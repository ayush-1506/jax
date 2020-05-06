# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implementation of pmap and related functionality.

Note on ShardingSpecs and spec_to_indices():
A ShardingSpec describes at a high level how a logical array is sharded across
devices (each ShardedDeviceArray has a ShardingSpec, and ShardingSpecs also
describe how to shard inputs to a parallel computation). spec_to_indices()
encodes exactly how a given ShardingSpec is translated to device buffers,
i.e. how the sharded array is "laid out" across devices. Given a sequence of
devices, we shard the data across the devices in row-major order, with
replication treated as an extra inner dimension.

For example, given the logical data array [1, 2, 3, 4], if we were to partition
this array 4 ways with a replication factor of 2, for a total of 8 devices, the
data on each device would be: [1, 1], [2, 2], [3, 3], [4, 4].

This encoding is assumed by various parts of the system, e.g. generating
replica groups for collective operations.
"""

from collections import defaultdict, namedtuple
from contextlib import contextmanager
from itertools import product
import operator as op
import threading
from typing import (Any, Callable, Dict, List, Optional, Sequence, Set, Tuple,
                    Type, Union)

from absl import logging
import numpy as onp

from ..config import flags
from .. import core
from .. import linear_util as lu
from .. import lazy
from ..core import Var, Literal
from ..abstract_arrays import (ConcreteArray, ShapedArray, array_types,
                               raise_to_shaped)
from ..util import (partial, unzip2, unzip3, prod, safe_map, safe_zip,
                    extend_name_stack, wrap_name)
from ..lib import xla_bridge as xb
from ..lib import xla_client as xc
from ..tree_util import tree_map
from .batching import broadcast, not_mapped, moveaxis
from . import batching
from . import partial_eval as pe
from . import xla
from . import ad


xops = xc.ops

FLAGS = flags.FLAGS

unsafe_map = map
map = safe_map

Index = Union[int, slice, Tuple[Union[int, slice], ...]]


# TODO(skye): make this a namedtuple. This may allow us to use ShardingSpecs in
# performance-sensitive code, e.g. shard_args.
class ShardingSpec:
  """Describes how a logical array is sharded across devices.

  Note this does not specify the physical devices to be sharded across, nor a
  logical ordering of data shards. Use `spec_to_indices` to resolve a
  ShardingSpec to the specific logical ordering expected throughout the system.

  Attributes:
    shards_per_axis: a tuple the same length as the array shape. Indicates how
      many shards each axis is divided into. Each axis must be divided into
      equal-sized shards (i.e. array_shape[i] % shards_per_axis[i] == 0).
    is_axis_materialized: a tuple the same length as the array shape. Indicates
      whether each axis of the array is represented in the on-device shape
      (i.e. sum(is_axis_materialized) == len(device_buffer.shape())). Any
      unmaterialized axes must be sharded into size-1 chunks
      (i.e. array_shape[i] == shards_per_axis[i]).
    replication_factor: the number of copies of the logical array.
  """
  def __init__(self,
               shards_per_axis: Tuple[int, ...],
               is_axis_materialized: Tuple[bool, ...],
               replication_factor: int):
    assert len(shards_per_axis) == len(is_axis_materialized)
    self.shards_per_axis = shards_per_axis
    self.is_axis_materialized = is_axis_materialized
    self.replication_factor = replication_factor

  def __eq__(self, other):
    return (self.shards_per_axis == other.shards_per_axis and
            self.is_axis_materialized == other.is_axis_materialized and
            self.replication_factor == other.replication_factor)

  def __repr__(self):
    return ("ShardingSpec(shards_per_axis=%s, is_axis_materialized=%s, "
            "replication_factor=%s)" %
            (self.shards_per_axis, self.is_axis_materialized,
             self.replication_factor))


def spec_to_indices(shape: Tuple[int, ...],
                    sharding_spec: ShardingSpec) -> Tuple[Index, ...]:
  """Returns numpy-style indices corresponding to sharding_spec.

  Each index describes a shard of the array. The order of the indices is the
  same as the device_buffers of a ShardedDeviceArray (i.e. the data is laid out
  row-major, with replication treated as an extra innermost dimension).

  Args:
    shape: The shape of the logical array being sharded.
    sharding_spec: Describes how the array is sharded.

  Returns:
    A tuple of length `prod(sharding_spec.shards_per_axis) *
    sharding_spec.replication_factor`.  Each element is an int, a slice object
    with step=1, or a tuple thereof, to be treated as an index into the full
    logical array.
  """
  assert len(shape) == len(sharding_spec.shards_per_axis), \
      f'{len(shape)} != {len(sharding_spec.shards_per_axis)}'
  indices_per_axis = [
      _axis_indices(axis_size, num_shards, is_materialized)
      for axis_size, num_shards, is_materialized in zip(
          shape, sharding_spec.shards_per_axis, sharding_spec.is_axis_materialized)]

  # Remove trailing slice(None) indices. This simplifies the final indices and
  # hopefully makes them more likely to match what's passed in by the user in
  # ShardedDeviceArray.__getitem__.
  while len(indices_per_axis) > 1 and indices_per_axis[-1] == [slice(None)]:
    indices_per_axis.pop(-1)

  # `product` will always return a sequence of tuples. Skip the tuples if each
  # index is a single element.
  if len(indices_per_axis) == 1:
    indices = list(indices_per_axis[0])
  else:
    indices = list(product(*indices_per_axis))

  return tuple(i for i in indices
               for _ in range(sharding_spec.replication_factor))


def _axis_indices(axis_size, num_shards, is_materialized):
  if not is_materialized:
    assert axis_size == num_shards, f'{axis_size} != {num_shards}'
    return list(range(axis_size))
  if num_shards == 1:
    return [slice(None)]
  shard_size, ragged = divmod(axis_size, num_shards)
  assert not ragged
  return [slice(i * shard_size, (i + 1) * shard_size) for i in range(num_shards)]


### util

def identity(x): return x

# TODO(skye): expose PyLocalBuffers in xla_client
def shard_args(devices: Sequence[xb.xla_client.Device],
               indices: Sequence[Sequence[Index]],
               args) -> Sequence[Sequence[xb.xla_client._xla.PyLocalBuffer]]:
  """Shard each argument data array along its leading axis.

  Args:
    devices: sequence of Devices mapping replica index to a physical device.
    indices: sequence of the same length as `args` describing how each arg
      should be sharded/replicated across `devices`. Each element in `indices`
      is the same length as `devices`.
    args: a sequence of JaxTypes representing arguments to be sharded according
      to `indices` and placed on `devices`.

  Returns:
    A list of device buffers with the same length as `devices` indexed by
    replica number, so that the nth element is the argument to be passed to the
    nth replica.
  """
  nargs, nrep = len(args), len(devices)
  buffers = [[None] * nargs for _ in range(nrep)]
  for a, arg in enumerate(args):
    # The shard_arg_handlers allow an extensible set of types to be sharded, but
    # inline handling for ShardedDeviceArray as a special case for performance
    # NOTE: we compare indices instead of sharding_spec because
    # pmap_benchmark.pmap_shard_args_benchmark indicates this is faster.
    if type(arg) is ShardedDeviceArray and indices[a] == arg.indices:
      for r, buf in enumerate(arg.device_buffers):
        buffers[r][a] = (buf if buf.device() == devices[r]
                         else buf.copy_to_device(devices[r]))
    else:
      arg = xla.canonicalize_dtype(arg)
      bufs = shard_arg_handlers[type(arg)](arg, devices, indices[a])
      for r, buf in enumerate(bufs):
        buffers[r][a] = buf

  return buffers


shard_arg_handlers: Dict[Any, Callable[[Any, Any, Any], Sequence[Any]]] = {}
shard_arg_handlers[core.Unit] = \
    lambda x, devices, _: [xla.device_put(core.unit, d) for d in devices]
def _shard_array(x, devices, indices):
  return [xla.device_put(x[i], d) for (i, d) in zip(indices, devices)]
for _t in array_types:
  shard_arg_handlers[_t] = _shard_array

def _shard_device_array(x, devices, indices):
  start_indices, limit_indices, removed_dims = map(tuple, unzip3(
      _as_slice_indices(x, idx) for idx in indices))
  shards = x._multi_slice(start_indices, limit_indices, removed_dims)
  return [xla.device_put(s, d) for s, d in zip(shards, devices)]
shard_arg_handlers[xla.DeviceArray] = _shard_device_array

# NOTE(skye): we could refactor to generate _multi_slice parameters directly
# from the input ShardingSpec, rather than the indices. However, this would
# require duplicating the ordering logic of spec_to_indices, which is more
# subtle and more likely to change than the index logic we have to support here.
def _as_slice_indices(arr: xla.DeviceArray, idx: Index) -> Tuple[
    Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
  """Returns start_indices, limit_indices, removed_dims"""
  start_indices = [0] * arr.ndim
  limit_indices = list(arr.shape)
  removed_dims = []

  tuple_idx = idx if isinstance(idx, tuple) else (idx,)
  for dim, sub_idx in enumerate(tuple_idx):
    if isinstance(sub_idx, int):
      start_indices[dim] = sub_idx
      limit_indices[dim] = sub_idx + 1
      removed_dims.append(dim)
    elif sub_idx == slice(None):
      continue
    else:
      assert isinstance(sub_idx, slice)
      assert isinstance(sub_idx.start, int)
      assert isinstance(sub_idx.stop, int)
      start_indices[dim] = sub_idx.start
      limit_indices[dim] = sub_idx.stop

  return tuple(start_indices), tuple(limit_indices), tuple(removed_dims) # type: ignore


def shard_aval(size, aval):
  try:
    return shard_aval_handlers[type(aval)](size, aval)
  except KeyError as err:
    raise TypeError("No shard_aval handler for type: {}".format(type(aval))
                    ) from err
shard_aval_handlers: Dict[Type[core.AbstractValue], Callable[[int, Any], Any]] = {}
shard_aval_handlers[core.AbstractUnit] = lambda size, x: x
def _shard_abstract_array(size, x):
  if not x.shape:
    raise ValueError("Scalar cannot be split across {} shards.".format(size))
  if x.shape[0] != size:
    raise ValueError("Axis size {} does not match leading dimension of "
                     "shape {}".format(size, x.shape))
  return ShapedArray(x.shape[1:], x.dtype)
shard_aval_handlers[ShapedArray] = _shard_abstract_array

# TODO(skye): expose PyLocalBuffers in xla_client
def aval_to_result_handler(size: int, nrep: int, aval: core.AbstractValue) \
    -> Callable[[List[xb.xla_client._xla.PyLocalBuffer]], Any]:
  if aval is not core.abstract_unit:
    sharding_spec = _pmap_sharding_spec(nrep, size, aval, True)
    indices = spec_to_indices((size,) + aval.shape, sharding_spec)
  else:
    sharding_spec = indices = None
  try:
    return pxla_result_handlers[type(aval)](size, sharding_spec, indices, aval)
  except KeyError as err:
    raise TypeError("No pxla_result_handler for type: {}".format(type(aval))
                    ) from err
PxlaResultHandler = Callable[..., Callable[[List[xb.xla_client._xla.PyLocalBuffer]], Any]]
pxla_result_handlers: Dict[Type[core.AbstractValue], PxlaResultHandler] = {}
pxla_result_handlers[core.AbstractUnit] = lambda *_: lambda _: core.unit
def array_result_handler(size, sharding_spec, indices, aval: ShapedArray):
  full_aval = ShapedArray((size,) + aval.shape, aval.dtype)
  return lambda bufs: ShardedDeviceArray(full_aval, sharding_spec, bufs,
                                         indices)
pxla_result_handlers[ShapedArray] = array_result_handler
pxla_result_handlers[ConcreteArray] = array_result_handler


### lazy device-memory persistence and result handling

class ShardedDeviceArray(xla.DeviceArray):
  """A ShardedDeviceArray is an ndarray sharded across devices.

  The purpose of a ShardedDeviceArray is to reduce the number of transfers when
  executing replicated computations, by allowing results to persist on the
  devices that produced them. That way dispatching a similarly replicated
  computation that consumes the same sharded memory layout does not incur any
  transfers.

  A ShardedDeviceArray represents one logical ndarray value, and simulates the
  behavior of an ndarray so that it can be treated by user code as an ndarray;
  that is, it is only an optimization to reduce transfers.

  Attributes:
    aval: A ShapedArray indicating the shape and dtype of this array.
    sharding_spec: describes how this array is sharded across `device_buffers`.
    device_buffers: the buffers containing the data for this array. Each buffer
      is the same shape and on a different device. Buffers are in row-major
      order, with replication treated as an extra innermost dimension.
    indices: the result of spec_to_indices(sharding_spec). Can optionally be
      precomputed for efficiency. A list the same length as
      `device_buffers`. Each index indicates what portion of the full array is
      stored in the corresponding device buffer, i.e. `array[indices[i]] ==
      device_buffers[i].to_py()`.
  """
  __slots__ = ["device_buffers", "sharding_spec", "indices"]

  # TODO(skye): expose PyLocalBuffers in xla_client
  def __init__(self,
               aval: ShapedArray,
               sharding_spec, # TODO(skye): add type annotation back, see below
               device_buffers: List[xb.xla_client._xla.PyLocalBuffer] = None,
               indices: Optional[Tuple[Index, ...]] = None):
    # TODO(skye): this is temporary staging while we switch users over to
    # providing sharding_spec. It assumes that any pre-existing callers are
    # creating pmap-style ShardedDeviceArrays.
    if device_buffers is None:
      device_buffers = sharding_spec
      sharded_aval = ShapedArray(aval.shape[1:], aval.dtype)
      sharding_spec = _pmap_sharding_spec(aval.shape[0], aval.shape[0],
                                          sharded_aval, True)

    # TODO(skye): assert invariants. Keep performance in mind though.
    if indices is None:
      indices = spec_to_indices(aval.shape, sharding_spec)
    self.aval = aval
    self.device_buffers = device_buffers
    self.sharding_spec = sharding_spec
    self.indices = indices
    self._npy_value = None
    if not core.skip_checks:
      assert type(aval) is ShapedArray

  def copy_to_host_async(self):
    if self._npy_value is None:
      # TODO(skye): only transfer one replica?
      for buf in self.device_buffers:
        buf.copy_to_host_async()

  def delete(self):
    for buf in self.device_buffers:
      buf.delete()
    self.device_buffers = None
    self._npy_value = None

  def _check_if_deleted(self):
    if self.device_buffers is None:
      raise ValueError("ShardedDeviceArray has been deleted.")

  def block_until_ready(self):
    self._check_if_deleted()
    for buf in self.device_buffers:
      buf.block_host_until_ready()
    return self

  @property
  def _value(self):
    if self._npy_value is None:
      self.copy_to_host_async()
      npy_value = onp.empty(self.aval.shape, self.aval.dtype)
      for i in range(0, len(self.device_buffers),
                     self.sharding_spec.replication_factor):
        npy_value[self.indices[i]] = self.device_buffers[i].to_py()
      self._npy_value = npy_value
    return self._npy_value

  def __getitem__(self, idx):
    if self._npy_value is None and idx in self.indices:
      buf = self.device_buffers[self.indices.index(idx)]
      aval = ShapedArray(buf.shape().dimensions(), self.aval.dtype)
      return xla.DeviceArray(aval, None, lazy.array(aval.shape), buf)
    else:
      return super(ShardedDeviceArray, self).__getitem__(idx)


def _hashable_index(idx):
  return tree_map(lambda x: (x.start, x.stop) if type(x) == slice else x,
                  idx)

# The fast path is handled directly in shard_args().
# TODO(skye): is there a simpler way to rewrite this using sharding_spec?
def _shard_sharded_device_array_slow_path(x, devices, indices):
  candidates = defaultdict(list)
  for buf, idx in zip(x.device_buffers, x.indices):
    candidates[_hashable_index(idx)].append(buf)

  bufs = []
  for idx, device in safe_zip(indices, devices):
    # Look up all buffers that contain the correct slice of the logical array.
    candidates_list = candidates[_hashable_index(idx)]
    if not candidates_list:
      # This array isn't sharded correctly. Reshard it via host roundtrip.
      # TODO(skye): more efficient reshard?
      return shard_arg_handlers[type(x._value)](x._value, devices, indices)
    # Try to find a candidate buffer already on the correct device,
    # otherwise copy one of them.
    for buf in candidates_list:
      if buf.device() == device:
        bufs.append(buf)
        break
    else:
      bufs.append(buf.copy_to_device(device))
  return bufs
shard_arg_handlers[ShardedDeviceArray] = _shard_sharded_device_array_slow_path

def _sharded_device_array_constant_handler(c, val, canonicalize_types=True):
  return xb.constant(c, onp.asarray(val), canonicalize_types=canonicalize_types)
xb.register_constant_handler(ShardedDeviceArray, _sharded_device_array_constant_handler)

core.pytype_aval_mappings[ShardedDeviceArray] = ConcreteArray
xla.device_put_handlers[ShardedDeviceArray] = xla._device_put_array
xla.pytype_aval_mappings[ShardedDeviceArray] = op.attrgetter('aval')
xla.canonicalize_dtype_handlers[ShardedDeviceArray] = identity


### the xla_pmap primitive and its rules are comparable to xla_call in xla.py

def xla_pmap_impl(fun: lu.WrappedFun, *args, backend, axis_name, axis_size,
                  global_axis_size, devices, name, mapped_invars):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = parallel_callable(fun, backend, axis_name, axis_size,
                                   global_axis_size, devices, name, mapped_invars,
                                   *abstract_args)
  return compiled_fun(*args)

@lu.cache
def parallel_callable(fun, backend, axis_name, axis_size, global_axis_size,
                      devices, name, mapped_invars, *avals):
  if devices is not None and len(devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  # Determine global_axis_size for use in AxisEnv.
  if devices:
    assert global_axis_size is None  # Checked in api.py
    global_axis_size = len(devices)
  elif xb.host_count() > 1:
    if global_axis_size is None:
      # TODO(skye): relax this constraint or provide functionality for
      # automatically passing appropriate `devices`.
      # TODO(trevorcai): This check forces us to provide global_axis_size for
      # all pmaps on pmap-on-pod. Can we do it after tracing?
      if axis_size != xb.local_device_count():
        raise ValueError(
            "On multi-host platforms, the input to pmapped functions must have "
            "leading axis size equal to the number of local devices if no "
            "`devices` argument is specified. Got axis_size=%d, "
            "num_local_devices=%d" % (axis_size, xb.local_device_count()))
      global_axis_size = xb.device_count()
  else:
    if global_axis_size is not None:
      if global_axis_size != axis_size:
        raise ValueError(
            "Specified axis_size {} doesn't match received axis_size {}.".format(
                global_axis_size, axis_size))
    else:
      global_axis_size = axis_size

  log_priority = logging.WARNING if FLAGS.jax_log_compiles else logging.DEBUG
  logging.log(log_priority,
              "Compiling {} for {} devices with args {}.".format(
                  fun.__name__, global_axis_size, avals))

  if devices:
    local_devices = [d for d in devices if d.host_id == xb.host_id()]
    assert len(local_devices) > 0
  else:
    local_devices = None

  sharded_avals = tuple(shard_aval(axis_size, aval) if m else aval
                        for m, aval in zip(mapped_invars, avals))
  with core.extend_axis_env(axis_name, axis_size):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, sharded_avals)
  jaxpr, uses_outfeed = xla.apply_outfeed_rewriter(jaxpr)

  # TODO(skye,mattjj): allow more collectives on multi-host as we test them, but
  # for now raise an error
  if devices is not None:
    is_multi_host_pmap = any(d.host_id != xb.host_id() for d in devices)
  else:
    is_multi_host_pmap = xb.host_count() > 1
  if is_multi_host_pmap:
    used_collectives = set(xla.jaxpr_collectives(jaxpr))
    if not used_collectives.issubset(multi_host_supported_collectives):
      msg = "using collectives that aren't supported for multi-host: {}"
      raise TypeError(msg.format(", ".join(map(str, used_collectives))))

  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  num_local_replicas = axis_size * jaxpr_replicas
  num_global_replicas = global_axis_size * jaxpr_replicas
  axis_env = xla.AxisEnv(num_global_replicas, (axis_name,), (global_axis_size,),
                         devices)

  handle_outs = avals_to_results_handler(axis_size, num_local_replicas, out_avals)

  if devices is None:
    if num_global_replicas > xb.device_count(backend):
      msg = ("compiling computation that requires {} replicas, but only {} XLA "
             "devices are available")
      raise ValueError(msg.format(num_global_replicas, xb.device_count(backend)))

    # On a single host, we use the platform's default device assignment to
    # potentially take advantage of device locality. On multiple hosts, the
    # default device assignment may interleave different hosts' replicas,
    # violating pmap's semantics where data is sharded across replicas in
    # row-major order. Instead, manually create a device assignment that ensures
    # each host is responsible for a continguous set of replicas.
    if num_global_replicas > num_local_replicas:
      # TODO(skye): use a locality-aware assignment that satisfies the above
      # constraint.
      devices = [d for host_id in xb.host_ids()
                 for d in xb.local_devices(host_id)]
    else:
      devices = xb.get_backend(backend).get_default_device_assignment(num_global_replicas)
  else:
    if num_local_replicas != len(local_devices):
      local_devices_str = ", ".join(map(str, local_devices))
      raise ValueError(
          "Leading axis size of input to pmapped function must equal the "
          "number of local devices passed to pmap. Got axis_size=%d, "
          "num_local_devices=%d.\n(Local devices passed to pmap: %s)"
          % (axis_size, len(local_devices), local_devices_str))
    if num_global_replicas != len(devices):
      raise ValueError("compiling computation that requires %s replicas, "
                       "but %s devices were specified"
                       % (num_global_replicas, len(devices)))

  tuple_args = len(sharded_avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  xla_args = xla._xla_callable_args(c, sharded_avals, tuple_args)
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, backend, axis_env, xla_consts,
                                extend_name_stack(wrap_name(name, 'pmap')), *xla_args)
  built = c.Build(xops.Tuple(c, out_nodes))

  device_assignment = tuple(d.id for d in devices)
  compile_options = xb.get_compile_options(
          num_replicas=num_global_replicas,
          num_partitions=1,
          device_assignment=device_assignment)
  compile_options.tuple_arguments = tuple_args
  backend = xb.get_backend(backend)
  compiled = backend.compile(built, compile_options=compile_options)

  input_sharding_specs = [_pmap_sharding_spec(num_local_replicas, axis_size,
                                              aval, m)
                          for m, aval in zip(mapped_invars, sharded_avals)]
  input_indices = [spec_to_indices(aval.shape, spec)
                   if spec is not None else None
                   for aval, spec in zip(avals, input_sharding_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)

  return partial(execute_replicated, compiled, uses_outfeed, backend, handle_args,
                 handle_outs)

multi_host_supported_collectives: Set[core.Primitive] = set()

class ResultToPopulate(object): pass
result_to_populate = ResultToPopulate()

def avals_to_results_handler(size, nrep, out_avals):
  nouts = len(out_avals)
  handlers = [aval_to_result_handler(size, nrep, aval) for aval in out_avals]
  def handler(out_bufs):
    buffers = [[result_to_populate] * nrep for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def replicate(val, axis_size, nrep, devices=None, backend=None):
  """Replicates ``val`` across multiple devices.

  Args:
    val: the value to be replicated.
    axis_size: the length of the output, i.e. the logical number of replicas to
    create. Usually equal to `nrep`, but in the case of nested pmaps, `nrep` may
    be a multiple of `axis_size`.
    nrep: the number of replicas to create. If ``devices`` is set, must be equal
      to ``len(devices)``.
    devices: the devices to replicate across. If None, ``nrep`` will be used to
      generate a default device assignment.
    backend: string specifying which backend to use.

  Returns:
    A ShardedDeviceArray of length `axis_size` where each shard is equal to
    ``val``.
  """
  device_count = (len(devices) if devices else xb.local_device_count())
  if nrep > device_count:
    msg = ("Cannot replicate across %d replicas because only %d local devices "
           "are available." % (nrep, device_count))
    if devices:
      msg += (" (local devices = %s)"
              % ", ".join(map(str, devices)) if devices else str(None))
    raise ValueError(msg)

  if devices is None:
    assert nrep is not None
    devices = xb.get_backend(backend).get_default_device_assignment(nrep)
  assert nrep == len(devices)

  # TODO(jekbradbury): use ShardingSpec.replication_factor instead
  aval = xla.abstractify(val)  # type: ShapedArray
  replicated_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
  sharding_spec = _pmap_sharding_spec(nrep, axis_size, aval, True)
  device_buffers = [xla.device_put(val, d) for d in devices]
  return ShardedDeviceArray(replicated_aval, sharding_spec, device_buffers)


def _pmap_sharding_spec(nrep, axis_size, sharded_aval, mapped):
  if sharded_aval is core.abstract_unit:
    return None
  replication_factor, ragged = divmod(nrep, axis_size)
  assert not ragged
  if mapped:
    return ShardingSpec(
        shards_per_axis=(axis_size,) + (1,) * len(sharded_aval.shape),
        is_axis_materialized=(False,) + (True,) * len(sharded_aval.shape),
        replication_factor=replication_factor)
  else:
    return ShardingSpec(
        shards_per_axis=(1,) * len(sharded_aval.shape),
        is_axis_materialized=(True,) * len(sharded_aval.shape),
        replication_factor=replication_factor * axis_size)


def execute_replicated(compiled, uses_outfeed, backend, in_handler, out_handler, *args):
  xla.check_before_outfeed_execution(uses_outfeed)
  input_bufs = in_handler(args)
  out_bufs = compiled.ExecuteOnLocalDevices(list(input_bufs))
  return out_handler(out_bufs)


xla_pmap_p = core.MapPrimitive('xla_pmap')
xla_pmap = xla_pmap_p.bind
xla_pmap_p.def_impl(xla_pmap_impl)

def _pmap_translation_rule(c, axis_env,
                           in_nodes, name_stack, axis_name, axis_size,
                           global_axis_size, devices, name,
                           call_jaxpr, *, backend=None, mapped_invars):
  # We in-line here rather than generating a Call HLO as in the xla_call
  # translation rule just because the extra tuple stuff is a pain.
  if axis_env.devices is not None or (axis_env.names and devices is not None):
    raise ValueError("Nested pmaps with explicit devices argument.")
  if global_axis_size is None:
    global_axis_size = axis_size
  new_env = xla.extend_axis_env(axis_env, axis_name, global_axis_size)
  # Shard the in_nodes that are mapped
  in_avals = [v.aval for v in call_jaxpr.invars]
  in_nodes_sharded = (
    _xla_shard(c, aval, new_env, in_node) if in_node_mapped else in_node
    for aval, in_node, in_node_mapped in zip(in_avals, in_nodes, mapped_invars))

  sharded_outs = xla.jaxpr_subcomp(
      c, call_jaxpr, backend, new_env, (),
      extend_name_stack(name_stack, wrap_name(name, 'pmap')), *in_nodes_sharded)
  out_avals = [v.aval for v in call_jaxpr.outvars]
  outs = [_xla_unshard(c, aval, new_env, shard, backend=backend)
          for aval, shard in zip(out_avals, sharded_outs)]
  return xops.Tuple(c, outs)

xla.call_translations[xla_pmap_p] = _pmap_translation_rule
ad.primitive_transposes[xla_pmap_p] = partial(ad.map_transpose, xla_pmap_p)

def _xla_shard(c, aval, axis_env, x):
  if aval is core.abstract_unit:
    return x
  elif isinstance(aval, ShapedArray):
    dims = list(c.GetShape(x).dimensions())
    zero = xb.constant(c, onp.zeros((), dtype=onp.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * (len(dims) - 1)
    return xops.Reshape(xops.DynamicSlice(x, idxs, [1] + dims[1:]), dims[1:])
  else:
    raise TypeError((aval, c.GetShape(x)))

# TODO(b/110096942): more efficient gather
def _xla_unshard(c, aval, axis_env, x, backend):
  if aval is core.abstract_unit:
    return x
  elif isinstance(aval, ShapedArray):
    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    convert_bool = (onp.issubdtype(aval.dtype, onp.bool_)
                    and xb.get_backend(backend).platform in ('cpu', 'gpu'))
    if convert_bool:
      x = xops.ConvertElementType(x, xb.dtype_to_etype(onp.float32))

    xla_shape = c.GetShape(x)
    dims = list(xla_shape.dimensions())
    padded = xops.Broadcast(xb.constant(c, onp.array(0, xla_shape.numpy_dtype())),
                         [axis_env.sizes[-1]] + dims)
    zero = xb.constant(c, onp.zeros((), dtype=onp.uint32))
    idxs = [_unravel_index(c, axis_env)] + [zero] * len(dims)
    padded = xops.DynamicUpdateSlice(padded, xops.Reshape(x, [1] + dims), idxs)
    replica_groups_protos = xc.make_replica_groups(
      xla.axis_groups(axis_env, axis_env.names[-1]))
    out = xops.CrossReplicaSum(padded, replica_groups_protos)

    # TODO(mattjj): remove this logic when AllReduce PRED supported on CPU / GPU
    if convert_bool:
      nonzero = xops.Ne(out, xb.constant(c, onp.array(0, dtype=onp.float32)))
      out = xops.ConvertElementType(nonzero, xb.dtype_to_etype(onp.bool_))
    return out
  else:
    raise TypeError((aval, c.GetShape(x)))

def _unravel_index(c, axis_env):
  div = xb.constant(c, onp.array(axis_env.nreps // prod(axis_env.sizes), onp.uint32))
  mod = xb.constant(c, onp.array(axis_env.sizes[-1], onp.uint32))
  return xops.Rem(xops.Div(xops.ReplicaId(c), div), mod)


def soft_pmap_impl(fun: lu.WrappedFun, *args, axis_name, axis_size, mapped_invars):
  abstract_args = unsafe_map(xla.abstractify, args)
  compiled_fun = _soft_pmap_callable(fun, axis_name, axis_size, mapped_invars,
                                     *abstract_args)
  return compiled_fun(*args)

@lu.cache
def _soft_pmap_callable(fun, axis_name, axis_size, mapped_invars, *avals):
  mapped_avals = [_mapped_aval(axis_size, aval) if m else aval
                  for m, aval in zip(mapped_invars, avals)]
  with core.extend_axis_env(axis_name, axis_size):
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, mapped_avals)
  jaxpr, uses_outfeed = xla.apply_outfeed_rewriter(jaxpr)

  num_devices = xb.local_device_count()
  chunk_size, ragged = divmod(axis_size, num_devices)
  if ragged:
    msg = f"number of devices {num_devices} must divide axis size {axis_size}"
    raise NotImplementedError(msg)

  jaxpr, _, consts = _soft_pmap_jaxpr(jaxpr, consts, mapped_invars,
                                      axis_name, chunk_size)
  jaxpr_replicas = xla.jaxpr_replicas(jaxpr)
  if jaxpr_replicas != 1: raise NotImplementedError

  tuple_args = len(avals) > 100  # pass long arg lists as tuple for TPU

  c = xb.make_computation_builder("soft_pmap_{}".format(fun.__name__))
  xla_consts = map(partial(xb.constant, c), consts)
  chunked_avals = [_unmapped_aval(chunk_size, aval) if m else aval
                   for m, aval in zip(mapped_invars, mapped_avals)]
  xla_args = xla._xla_callable_args(c, chunked_avals, tuple_args)
  axis_env = xla.AxisEnv(num_devices, (axis_name,), (num_devices,), None)
  out_nodes = xla.jaxpr_subcomp(c, jaxpr, None, axis_env, xla_consts,
                                'soft_pmap', *xla_args)
  built = c.Build(xops.Tuple(c, out_nodes))

  compile_options = xb.get_compile_options(
          num_replicas=num_devices, num_partitions=1, device_assignment=None)
  compile_options.tuple_arguments = tuple_args
  backend = xb.get_backend(None)
  compiled = backend.compile(built, compile_options=compile_options)

  input_specs = [
      ShardingSpec(shards_per_axis=(num_devices,) + (1,) * (aval.ndim - 1),
                   is_axis_materialized=(True,) * aval.ndim,
                   replication_factor=1)
      if mapped else
      ShardingSpec(shards_per_axis=(1,) * aval.ndim,
                   is_axis_materialized=(False,) + (True,) * (aval.ndim - 1),
                   replication_factor=num_devices)
      for aval, mapped in zip(avals, mapped_invars)]
  input_indices = [spec and spec_to_indices(aval.shape, spec)
                   for aval, spec in zip(avals, input_specs)]
  handle_args = partial(shard_args, compiled.local_devices(), input_indices)
  handle_outs = soft_pmap_avals_to_results_handler(num_devices, chunk_size, out_avals)

  return partial(execute_replicated, compiled, uses_outfeed, backend,
                 handle_args, handle_outs)

def _soft_pmap_jaxpr(jaxpr, consts, mapped_invars, axis_name, chunk_size):
  fun = partial(_soft_pmap_interp, chunk_size, jaxpr, consts, mapped_invars)
  in_avals = [_unmapped_aval(chunk_size, v.aval) if m else v.aval
              for v, m in zip(jaxpr.invars, mapped_invars)]
  return pe.trace_to_jaxpr_dynamic(lu.wrap_init(fun), in_avals)

def _soft_pmap_interp(chunk_size, jaxpr, consts, mapped_invars, *args):
  env: Dict[Var, Tuple[Any, bool]] = {}

  def read(atom: Union[Var, Literal]) -> Tuple[Any, bool]:
    if isinstance(atom, Literal):
      return (atom.val, False)
    else:
      return env[atom]

  def write(v: Var, val: Any, mapped: bool) -> None:
    env[v] = (val, mapped)

  write(core.unitvar, core.unit, False)
  map(write, jaxpr.constvars, consts, (False,) * len(consts))
  map(write, jaxpr.invars, args, mapped_invars)
  for eqn in jaxpr.eqns:
    in_vals, in_mapped = unzip2(map(read, eqn.invars))
    if eqn.primitive in xla.parallel_translations:
      rule = soft_pmap_rules[eqn.primitive]
      out_vals, out_mapped = rule(in_vals, in_mapped, chunk_size, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_mapped = [out_vals], [out_mapped]
    elif isinstance(eqn.primitive, core.CallPrimitive):
      # we just inline here for convenience
      call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
      out_vals = _soft_pmap_interp(chunk_size, call_jaxpr, (), in_mapped, *in_vals)
      out_mapped = [True] * len(out_vals)
    elif isinstance(eqn.primitive, core.MapPrimitive):
      raise NotImplementedError  # TODO
    else:
      rule = batching.get_primitive_batcher(eqn.primitive)
      in_axes = [0 if m else batching.not_mapped for m in in_mapped]
      out_vals, out_axes = rule(in_vals, in_axes, **eqn.params)
      if not eqn.primitive.multiple_results:
        out_vals, out_axes = [out_vals], [out_axes]
      out_vals = [moveaxis(x, d, 0) if d is not not_mapped and d != 0 else x
                  for x, d in zip(out_vals, out_axes)]
      out_mapped = [d is not not_mapped for d in out_axes]
    map(write, eqn.outvars, out_vals, out_mapped)

  out_vals, out_mapped = unzip2(map(read, jaxpr.outvars))
  out_vals = [out if mapped else broadcast(out, chunk_size, 0)
              for out, mapped in zip(out_vals, out_mapped)]
  return out_vals

# TODO dedup these functions with other aval_to_result_handler via ShardingSpec
def soft_pmap_avals_to_results_handler(num_devices, chunk_size, out_avals):
  nouts = len(out_avals)
  handlers = [soft_pmap_aval_to_result_handler(chunk_size, num_devices, aval)
              for aval in out_avals]
  def handler(out_bufs):
    buffers = [[result_to_populate] * num_devices for _ in range(nouts)]
    for r, tuple_buf in enumerate(out_bufs):
      for i, buf in enumerate(tuple_buf):
        buffers[i][r] = buf
    assert not any(buf is result_to_populate for bufs in buffers
                   for buf in bufs)
    return [h(bufs) for h, bufs in zip(handlers, buffers)]
  return handler

def soft_pmap_aval_to_result_handler(chunk_size, num_devices, aval):
  axis_size = chunk_size * num_devices
  if aval is core.abstract_unit:
    return lambda _: core.unit
  elif isinstance(aval, core.ShapedArray):
    new_aval = ShapedArray((axis_size,) + aval.shape, aval.dtype)
    spec = ShardingSpec(shards_per_axis=(num_devices,) + (1,) * aval.ndim,
                        is_axis_materialized=(True,) * new_aval.ndim,
                        replication_factor=1)
    return lambda bufs: ShardedDeviceArray(new_aval, spec, bufs)
  else:
    raise TypeError(aval)

# TODO use core.py versions on master
def _unmapped_aval(size, aval):
  if aval is core.abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    return ShapedArray((size,) + aval.shape, aval.dtype)
  else:
    raise TypeError(aval)

def _mapped_aval(size, aval):
  if aval is core.abstract_unit:
    return aval
  elif isinstance(aval, ShapedArray):
    # might be raising abstraction level from Concrete here
    assert aval.shape[0] == size
    return ShapedArray(aval.shape[1:], aval.dtype)
  else:
    raise TypeError(aval)

soft_pmap_p = core.MapPrimitive('soft_pmap')
soft_pmap = soft_pmap_p.bind
soft_pmap_p.def_impl(soft_pmap_impl)

soft_pmap_rules: Dict[core.Primitive, Callable] = {}


def _axis_index_soft_pmap_rule(vals, mapped, chunk_size, *, axis_name):
  assert not vals and not mapped
  idx = core.axis_index(axis_name)
  return idx * chunk_size + onp.arange(chunk_size), True
soft_pmap_rules[core.axis_index_p] = _axis_index_soft_pmap_rule
