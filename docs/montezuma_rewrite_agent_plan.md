---
document: Montezuma rewrite agent handoff and execution plan
audience: coding agents
baseline_commit: 0fc7635
baseline_branch: monte_ai
status: reconnaissance complete; no runtime implementation changes in this handoff
primary_goal: Replace room-specialized dispatch/persistence with fixed-shape JAX layout/state data while retaining programmatic authoring and sharing one immutable layout bundle across a batch.
---

# Montezuma rewrite — agent context and execution plan

## 0. Operating contract

This document is the source of truth for the rewrite. Read it before modifying
Montezuma code. It intentionally describes contracts, data ownership, and
verification rather than narrating implementation history.

### Required outcomes

1. Preserve fully programmatic layout definition. A Python-side builder must be
   able to define rooms, connectivity, collision rasters, visual layers, and
   feature instances without hand-writing a new JAX branch per room.
2. The hot simulation path must not raise a room-specific `NamedTuple`,
   serialize it, deserialize it, or dispatch across every room.
3. All runtime leaves must have fixed shape and dtype. Padded capacity plus a
   validity mask is the normal representation for optional/variable-count
   features.
4. `jax.jit`, `jax.vmap`, and `lax.scan` must work for both homogeneous- and
   mixed-**room-id** batches. Production batches deliberately share one static
   layout; only state and actions are batched.
5. Behavior must be protected by semantic regression tests before changing the
   implementation. Do not use the current internal state layout as the public
   golden format.
6. Keep simulation and rendering separately measurable. Pixel-mode performance
   is not evidence of simulation performance.

### Non-goals for the first rewrite

- Do not preserve SANTAH/proto-room/persistence internals for compatibility.
  They are implementation details and the principal source of dispatch.
- Do not silently change user-visible semantics while optimizing. Record an
  intentional deviation in the decision ledger and update the relevant tests.
- Do not make a capacity dynamically grow inside JIT. A new capacity signature
  means a new compiled shape; a new layout with the same signature must not
  require a new runtime schema (though the artifact-bound production cache may
  deliberately compile a new executable for its fingerprint).
- Do not optimize renderer output before a clean simulation reference exists.
- Do not require heterogeneous layouts inside one production `vmap` batch. Keep
  that only as an optional functional-core research/test mode; it is not needed
  by the JaxAtari integration contract for this project.

### First read order

1. `src/jaxatari/environment.py` — base environment contract.
2. `src/jaxatari/wrappers.py` — frame skip, vectorized wrappers, rendering
   entry points, and episodic-life behavior.
3. `src/jaxatari/games/jax_montezuma.py` — current simulation and renderer.
4. `src/jaxatari/games/jax_mzuma_enums_and_nts.py` — state/tag/entity schema.
5. `src/jaxatari/games/jax_mzuma_utils.py` — SANTAH/PyramidLayout machinery.
6. `src/jaxatari/games/jax_mzuma_layouts.py` — current programmatic layouts.
7. Sections 5–10 of this document before creating new runtime code.

## 1. Repository and framework map

| Area | Current responsibility | Rewrite implication |
| --- | --- | --- |
| `environment.py` | `JaxEnvironment` API: `reset`, `step`, `render`, spaces, observation flattening, reward/done/info | The public façade must continue to satisfy this contract. |
| `wrappers.py` | `AtariWrapper` frame skip/stacking/sticky action; object/pixel wrappers; logging | Test raw transition and wrappers separately. Default frame skip runs four raw steps. |
| `gym_wrapper.py` | Functional Gym adapter; requests rendered pixels | Pixel/Gym benchmarks include render cost. |
| `rendering/jax_rendering_utils.py` | generic `render_at` | Current helper performs full-frame mesh/grid/pad/gather/blend per stamp; do not assume it is cheap. |
| `jax_montezuma.py` | game state, mechanics, rendering, construction | Split into layout compiler, pure core, observation, renderer, façade. |
| `jax_mzuma_enums_and_nts.py` | current heterogeneous room/tag/entity `NamedTuple` definitions | Use as a semantic inventory, not as the target runtime schema. |
| `jax_mzuma_utils.py` | SANTAH serialization + PyramidLayout + room dispatch | Keep only temporarily for legacy-reference fixture generation. Remove from new hot path. |
| `jax_mzuma_layouts.py` | Python authoring of test/demo/difficulty layouts | Port to a new builder; this is the programmatic-layout contract to preserve. |
| `tests/` | generic tests and non-Montezuma snapshots | Add a dedicated Montezuma suite; existing snapshots are insufficient. |

### JAXAtari execution model

```text
Python construction / layout authoring
  -> JaxEnvironment instance (usually captured as static self)
  -> reset(key) -> (observation, immutable JAX PyTree state)
  -> step(state, action) -> (observation, new_state, reward, done, info)
  -> caller may apply jit, vmap, and scan

AtariWrapper
  -> sticky action + lax.scan(frame_skip) + auto-reset + frame stack
Object/Pixels wrapper
  -> object flattening and/or render(state)
```

JAX requires all carried PyTree leaves to have stable shape and dtype. This is
why heterogeneous rooms were introduced, but shape homogenization should happen
at layout compilation time, not by generating a run-time branch per room.

### Current integration facts

- Montezuma is instantiated directly as `JaxMontezuma(...)`; it is not in
  `jaxatari.core.GAME_MODULES` (`src/jaxatari/core.py`).
- `JaxMontezuma.__init__` accepts `frameskip`, but does not use it. Frame skip
  belongs to `AtariWrapper`.
- Montezuma is not represented in the current generic regression snapshots.
- Generic game discovery currently treats every `jax_*.py` file as a game,
  including helper modules. New test discovery must use an explicit registry or
  inspect for a `JaxEnvironment` subclass.

## 2. Current Montezuma architecture

### 2.1 Construction-time model

`JaxMontezuma.__init__` does the following:

1. Registers shared proto-room/tag metadata in the global `SANTAH` singleton.
2. Builds one layout through `PyramidLayout` and `make_*_layout` functions.
3. Builds a variable-width integer `persistence_state` with one row per room.
4. Generates loader/writer functions and a wrapper for each room-aware
   operation.
5. Optionally constructs a separate `MontezumaRenderer`, which rebuilds the
   layout and creates another set of room-dispatch wrappers.

Key source locations:

- Room/tag registration: `jax_montezuma.py:1149-1259`.
- Layout selection and wrapper installation: `jax_montezuma.py:1262-1348`.
- Global SANTAH state and proto-room design: `jax_mzuma_utils.py:48-132`.
- Python room creation API: `jax_mzuma_utils.py:1179-1224`.

`SANTAH` is process-global. Its `register_proto_room` returns early when a
second proto room is registered (`jax_mzuma_utils.py:155-162`), so it is not a
sound basis for independent layout construction/testing in one process.

### 2.2 Current per-step flow

```text
step
  -> frozen ? continue_freeze : _non_frozen_step
     -> player_step
        -> rebuild augmented collision map
        -> input, horizontal/vertical motion
        -> climbing and room-feature interactions
        -> convolution-based collision correction
     -> room transition / counters / fall damage
  -> freeze transition bookkeeping
  -> wrapped full observation construction
  -> reward, done, info
```

The normal non-frozen step has **17 room-dispatched sites**:

| Stage | Wrapped calls on the normal path |
| --- | --- |
| `player_step` | augment collision; bonus floor; doors; dropout floors; side walls; conveyors; conveyor movement; start climbing; stop climbing; pit; laser; items; door unlock; enemies; enemy collision (**15**) |
| `_non_frozen_step` | counter update (**1**) |
| `step` | observation construction (**1**) |

Room transitions add proto-room write/load and entrance callbacks. The renderer
has **12** analogous room-dispatched rendering stages.

### 2.3 Why dispatch is expensive

`PyramidLayout._wrap_lowered_function` (`jax_mzuma_utils.py:1286-1401`):

1. generates one specialized closure per room;
2. raises the generic proto room to that room's full `NamedTuple`;
3. invokes the room-specific routine;
4. serializes the room back to a persistence row;
5. dynamically updates the full persistence matrix;
6. reloads/lower the proto room; and
7. selects the closure with `jax.lax.switch(room_id, wrapped_functions, ...)`.

This happens even for operations that only read static room data. The function's
own docstring explicitly says its overhead is linear in room count.

Under `vmap`, a room-indexed `lax.switch` cannot be treated as one branch for a
whole batch when lanes differ. A local JAX 0.4.35 diagnostic showed the vmapped
switch represented as predicates/selects with all toy branch arithmetic in the
jaxpr. Treat mixed-room batch scaling as a required GPU measurement, not as an
assumption that branch code is skipped per warp.

### 2.4 Current state/data ownership

`MontezumaState` (`jax_mzuma_enums_and_nts.py:453-543`) combines:

- a proto `room_state` plus `persistence_state`;
- player/gameplay scalars;
- full-screen `augmented_collision_map`;
- mutable render `canvas`;
- an embedded full `observation`; and
- a nested `frozen_state` used to resume after freezes.

The embedded arrays are costly to carry, duplicate, and sometimes nest inside
the frozen state. A new state should contain simulation data only. Observation
and render buffers are outputs/scratch values, not carried gameplay state.

### 2.5 Coordinate conventions to preserve or deliberately adapt

Current simulation usually represents player/entity y coordinates **relative to
the current room**. It adds `room_state.vertical_offset` when indexing a
full-screen map. Collision arrays use an XY-oriented `[width, height]` order;
final rendered images are HWC and are transposed near the end of rendering.

The target engine should use one canonical full-screen coordinate system
internally, preferably `xy_screen` or (better for rendering) `yx_screen` with
the axis named everywhere. The layout compiler may accept room-local authoring
coordinates and shifts them once. Preserve legacy observation coordinate
semantics through an explicit adapter until a versioned API change is approved.

Never rely on an undocumented transpose. Give every target schema field an axis
comment, e.g. `[room, y, x]`, `[room, entity, xy]`.

## 3. Evidence-backed bottleneck inventory

| Priority | Evidence | Effect | Target response |
| --- | --- | --- | --- |
| P0 | 17 normal-step + 12 render room wrappers, each ultimately uses room-count `lax.switch` | Compile graph and mixed-batch work grow with room count | Gather/index fixed-shape arrays; zero hot-path room-count switches. |
| P0 | Every wrapper raises, serializes, updates persistence, and reloads proto room | Repeated typed reconstruction and writes dominate graph complexity | Store dynamic room fields directly in a homogeneous DynamicRoomsState pytree. |
| P0 | `find_nearest_free_position_2D_conv` uses full 160x210 `convolve2d` on every `fix_player_position` call | Full-playfield convolution in the ordinary motion path | Exact local collision queries plus fixed bounded swept resolution. |
| P1 | `player_step` rebuilds/ORs a full collision map through multiple wrappers | Full-screen memory traffic before a local collision query | Keep static raster collision in layout; query dynamic colliders directly. |
| P1 | Observation builder creates two full 160x210 maps and dispatches by room every step | Large, repeated object-observation work | Precompute static annotation per room and gather it. |
| P1 | `AtariWrapper` scans raw `step`; Pixel/functional paths discard many returned object observations | Frame skip can pay for observations that no consumer uses | Expose a transition-only path and construct observations only at the public boundary that needs them. |
| P1 | State carries canvas, collision map, observation, and freeze snapshot | Batch memory pressure and larger state traffic | Keep only gameplay state; create output scratch arrays on demand. |
| P1 | Renderer rebuilds layout and stamps using full-frame `render_at` | Pixel-wrapper throughput can be dominated by rendering | Gather precomposed static background and use bounded sprite compositing. |
| P2 | Current room arrays have variable dimensions | Original reason for proto dispatch | Pad exactly once in compiler with validity masks/counters. |

### Verified local diagnostics (directional only)

Environment caveats: the repository `.venv` has CPU JAX/JAXLIB 0.4.35, whereas
`pyproject.toml` pins 0.6.0; no CUDA/GPU, pytest, or pygame was available. These
numbers identify mechanisms, not release performance.

- A renderless one-room test-layout step first-JIT compiled in roughly 7.7–9.5
  seconds locally; warm cached raw steps were on the order of several ms.
- A renderless 24-room difficulty-1 layout took about 14.4 s to construct and
  2.9 s to compile reset. Its first step compilation did not complete cleanly
  in the diagnostic process. Treat that as a compilation/memory-risk signal,
  not a quantified GPU result.
- A five-room demo first-step attempt also terminated before a successful JIT
  result under local memory pressure; `JAX_DISABLE_JIT` could execute it.
- The full convolution collision helper had a local warm median of about
  3.24 ms; current `ray_cast_downwards` was about 0.012 ms in the same
  diagnostic. This is a concrete hotspot, not merely a hypothesis.

### Current difficulty-1 layout inventory

Host-side inspection of `difficulty_1` found 24 rooms and persistence shape
`[24, 21]`. Variable dimensions include:

| Field kind | Current observed capacities/shapes |
| --- | --- |
| room collision raster | `[160, 147]`, `[160, 148]`, `[160, 149]`, `[160, 150]`, `[160, 151]` |
| items / room | 1–3 |
| enemies / room | 1–2 |
| ladders / room | 1–3 |
| ropes / room | 1 or 3 |
| doors / room | 2 where present |
| laser barriers / room | 6 or 8 |
| dropout floors / room | 1 or 12 |
| vertical offset | 47 in all inspected difficulty-1 rooms |

Several derived maps are already full `[160, 210]`, but that partial padding
does not remove heterogeneity of room classes, tag field sets, entity stack
lengths, or persistence schemas. A historical non-baseline commit `d334bb5`
attempted room-sized field padding; audit it for useful preprocessing ideas, but
do not mistake it for a dispatch-free runtime design.

## 4. Legacy semantics that require an explicit decision

Create `tests/montezuma/DECISIONS.md` or an equivalent decision table before
porting any behavior. Each entry must say `preserve`, `fix`, or `defer`, name an
owner, and point to a regression test.

| Topic | Current behavior / source | Default recommendation |
| --- | --- | --- |
| Life field spelling | State uses `lifes`; `AtariWrapper(episodic_life=True)` only checks `lives`/`lives_lost` | Standardize new state on `lives` and test wrapper episodic life; record as intentional compatibility repair. |
| Dropout observation flag | Observation else branch sets `has_dropout_floor` to 1 (`jax_montezuma.py:2452-2454`) | Fix only with a named semantic test and release note. |
| Montezuma registration | absent from `jaxatari.make` | Add explicit construction/registration after new façade is stable. |
| Constants configuration | `LAYOUT`/`RENDERLESS` are unannotated class attributes of `MontezumaConstants` | Replace with an explicit immutable config/layout argument. |
| `self.LAYOUT` | constructor initializes it but builds into a local variable | Provide explicit inspectable compiled layout in new façade. |
| Freeze render semantics | pre-step state renders while post-step state resumes | Preserve initially; isolate as a compact FreezeState and golden-render test. |
| Player y coordinate | room-relative in legacy state/observation | Preserve output initially or version the observation API deliberately. |
| `RENDERLESS` | skips renderer and returns `None` from render | Define exact new renderless contract; do not let it change simulation/observation semantics. |

## 5. Target architecture

### 5.1 Separation of concerns

```text
host-only LayoutSpec / LayoutBuilder
  -> validation + capacity selection + raster/asset preprocessing
  -> versioned layout artifact: manifest + fixed-shape array bundle
  -> load once at environment construction; validate and device-place once
  -> immutable bound LayoutBundle, shared by every lane and the renderer

production bound JAX core (the normal JaxAtari path):
  reset_bound(key) -> SimState
  transition_bound(state, action) -> SimState
  observe_bound(state) -> MontezumaObservation
  render_bound(state) -> uint8[H, W, 3]
  # LayoutBundle is captured/bound, not a state or vmap argument.

optional functional/reference core (tests and future research only):
  reset_core(layout, key) -> SimState
  transition_core(layout, state, action) -> SimState
  observe(layout, state) -> MontezumaObservation
  render_core(layout, state) -> uint8[H, W, 3]

JaxMontezuma façade:
  owns one bound LayoutBundle and exposes the existing reset/step/render API
  vmap(step, in_axes=(0, 0)) maps state/action only; layout is shared
```

### 5.1.1 Production layout ownership and artifact contract

The preferred production design is a **single immutable layout bundle per
environment/batch**, owned by the game instance and shared with its renderer.
It is not a leaf of `SimState`, is never replicated into a batched state, and
is not passed with `in_axes=0` under `vmap`. This achieves the important win:
static collision/annotation/render data is stored and transferred once rather
than B times. It can also enable constant/precomputation benefits, although it
does not eliminate the necessary indexed gather for each lane's active room.

The builder must be able to emit a portable, non-pickle artifact, for example:

```text
layouts/<layout-fingerprint>/manifest.json   # schema version, capacity, axes,
                                               # asset hashes, builder version
layouts/<layout-fingerprint>/arrays.npz      # fixed-shape static arrays and
                                               # initial DynamicRoomsState template
```

`LayoutBundle.load(...)` runs on the host during `JaxMontezuma` construction,
validates the manifest/fingerprint, converts leaves to JAX arrays, and
device-places them once. There is no disk I/O, asset decoding, Python builder,
or artifact lookup inside `jit`, `scan`, or `vmap`. A programmatically created
`LayoutSpec` remains first-class: compile it in the host process, optionally
cache the same artifact, then bind the resulting bundle.

Changing layouts means constructing/binding a new environment (and renderer)
or selecting a cached bound bundle **between** runs. It is an intentional JIT
cache boundary keyed by layout fingerprint/capacity; a running batch never
switches layouts. This needs no general JaxAtari framework change because an
environment instance is already captured by its bound methods. The renderer
must receive the exact same `LayoutBundle`, never rebuild the layout.

Only if M7 demonstrates that an existing wrapper reconstructs or batches
environment-owned static data should the framework gain a narrow optional
`bind_static_context`/compile-key hook. Such a hook must retain the bundle on
the environment side, affect no other game by default, and must never add a
layout field to an environment state or wrapper PyTree.

Keep the explicit `*_core(layout, ...)` functions as a thin, testable internal
layer. They are useful for compiler tests and an optional future
heterogeneous-layout experiment, but are not the default training API. In that
mode, a layout is an ordinary unbatched PyTree parameter, never mutable global
state or a component of `SimState`.

Benchmark the bound-artifact and explicit-unbatched-parameter modes on the
target GPU before claiming a raw-step difference. The expected material win is
eliminating accidental B-fold layout/state storage and host-to-device traffic;
the disk file itself is a construction-time delivery mechanism, not a per-step
optimization.

Use `flax.struct.dataclass` (already a project dependency) or another registered
immutable PyTree type. Avoid generated runtime `NamedTuple` types, class-level
registries, and per-room JIT construction.

### 5.2 Capacity signature

```text
LayoutCapacity (Python/static, determines shapes)
  num_rooms: R
  max_items: I
  max_enemies: E
  max_ladders: L
  max_ropes: P
  max_doors: D
  max_barriers: B
  max_dropout_floors: F
  max_conveyors: C
  screen_height: H
  screen_width: W
```

The builder either infers exact maxima from a `LayoutSpec` or accepts declared
headroom. An overflow must fail at build time with the room id, feature kind,
actual count, and capacity. Inert padded entries always use a `valid=False`
mask; never give them a fake entity type that can participate in gameplay.

Same capacity + different values must retain identical PyTree shapes/dtypes.
Changing capacity intentionally creates a new compiled program. The production
artifact-bound API may also intentionally use a separate cached executable per
layout fingerprint; shape equality is still required for the functional-core
and compiler contracts.

Capacity metadata is static, but the large array-valued layout must not be made
a blanket `static_arg` to JIT. In production, bind/device-place the immutable
bundle once and benchmark its closure/bound-method lowering on the target
stack. In the explicit functional-core mode, pass layout leaves as ordinary
unbatched device-resident PyTree inputs. Neither mode puts layout leaves in the
batched state or relies on hashing a giant array PyTree as a static argument.

### 5.3 Proposed static layout schema

All axes below are normative examples. Pick `[room, y, x]` for screen rasters
and document it in the actual classes.

```text
MontezumaLayout
  capacity: non-array/static metadata (or separate host object)
  valid_room:              bool[R]
  room_y0:                 int16[R]
  room_height:             int16[R]
  spawn_xy:                int16[R, 4, 2]       # entry side order is fixed enum order
  neighbour_room:          int32[R, 4]          # -1 means no edge
  neighbour_entry_side:    int8[R, 4]
  room_flags:              bool[R, NUM_FLAGS]

  base_solid:              bool[R, H, W]        # full screen, static raster
  static_annotation:       int16[R, H, W, 2]    # precomputed legacy observation channels
  static_rgba:             uint8[R, H, W, 4]    # precomposed non-dynamic visual layer

  item_kind:               int8[R, I]
  item_initial_xy:         int16[R, I, 2]
  item_valid:              bool[R, I]
  door_xywh:               int16[R, D, 4]
  door_valid:              bool[R, D]
  ladder_geometry:         int16[R, L, ...]
  ladder_valid:            bool[R, L]
  rope_geometry:           int16[R, P, ...]
  rope_valid:              bool[R, P]
  enemy_initial:           compact static enemy parameters[R, E, ...]
  enemy_valid:             bool[R, E]
  barrier_geometry:        int16[R, B, 4]
  barrier_valid:           bool[R, B]
  dropout_geometry:        int16[R, F, ...]
  dropout_valid:           bool[R, F]
  conveyor_geometry_dir:   int16[R, C, ...]
  conveyor_valid:          bool[R, C]
  optional_static_masks:   only where a rectangle/table cannot represent a rule

  initial_rooms:           DynamicRoomsState initialization template
```

The schema must support arbitrary raster collision/static visual content, so
procedural authoring is not restricted to a fixed list of room templates.
Feature tables provide efficient interaction queries; optional masks are for
genuinely irregular dynamic geometry and are queried locally rather than ORed
into a full playfield every frame.

Padding has a field-specific fill policy: off-room collision pixels are
**solid**, ordinary masks are false, identifier maps use `-1`, and unused RGBA
pixels are transparent. Validity masks gate every table reduction/indexing;
sentinel values alone are never gameplay logic.

### 5.4 Proposed dynamic state schema

```text
SimState
  active_room:             int32[]
  player: PlayerState      # xy, velocity/input, motion, attachment, animation
  game: GameState          # frame, score, lives, inventory, rng, timers
  rooms: DynamicRoomsState
  freeze: FreezeState

DynamicRoomsState          # all leaves are homogeneous, indexed by room
  item_on_field:           bool[R, I]
  item_xy:                 int16[R, I, 2]       # mutable bonus item positions
  door_on_field:           bool[R, D]
  enemy: EnemyDynamicState # [R, E, ...] leaves: alive, xy, direction, counters
  barrier_cycle:           int32[R]
  bonus_cycle:             int32[R]
  rope_last_hung:          int16[R]
  ... only values that actually persist

FreezeState
  active, type, remaining
  resume_state or explicitly minimal gameplay snapshot
```

Do **not** put `[H, W]` augmented collision, render canvas, or full observation
inside `SimState`. Do not store a serialized generic buffer alongside direct
typed room data. If freeze fidelity needs a snapshot, verify what minimal
gameplay data is actually needed and keep rendering scratch out of it.

### 5.5 Hot-path rules

```python
room_id = state.active_room
room_static = jax.tree.map(lambda x: x[room_id], layout.room_indexed_fields)
room_dynamic = jax.tree.map(lambda x: x[room_id], state.rooms.room_indexed_fields)

# operate on padded [capacity, ...] arrays with valid masks
updated_room_dynamic, player, game = step_room(...)

# write only selected rows back once per dynamic group
new_rooms = state.rooms.replace(
    item_on_field=state.rooms.item_on_field.at[room_id].set(updated_room_dynamic.item_on_field),
    ...,
)
```

Gather the active dynamic row once, let all feature handlers update that small
active-row PyTree, then scatter it once at end of frame. A room transition may
need one additional target-row update; do not scatter the `[R, ...]` table after
each feature handler.

Required constraints:

- No `lax.switch(active_room, room_functions)` in simulation, observation, or
  renderer.
- No room-state serializer/deserializer in the hot path.
- A small switch over action or enemy kind is acceptable only if its branch count
  is a fixed game-rule constant, never a room count. Prefer table/mask arithmetic
  where it prevents vmapped all-branch work.
- Use vectorized padded entities plus deterministic masks/reductions. Preserve
  old priority rules explicitly (e.g. which of two simultaneous pickups wins).
- On a room transition, change `active_room`, gather the spawn record, and update
  entry state. Do not copy current room data into a generic persistence buffer.

### 5.6 Collision design

Keep exact static pixel collision but stop solving a local movement problem with
a global convolution.

1. `base_solid[room_id]` is a full-screen static raster, including out-of-room
   boundary solidity.
2. `collides_static(player_xy, hitbox)` uses a bounded dynamic slice/reduction
   against only the player hitbox.
3. Dynamic solid features (doors, active laser barriers, dropout floors,
   sidewalls, bonus floor, conveyors) are queried as padded rectangles/segments
   with masks. Use a local mask slice only for irregular cases.
4. Resolve movement with a fixed bounded sweep (one pixel/substep, maximum
   displacement derived from game constants), preserving one-way/jump semantics.
5. Keep specialized ladder/rope/conveyor queries as local feature-table queries.
6. Keep the existing down-ray logic initially if needed for behavior parity; it
   is much smaller than the 2-D convolution and can be optimized after parity.

The compiler is responsible for converting author-friendly room-local geometry
to the canonical coordinate system. The simulator never adds an ad hoc room
vertical offset to a map access.

### 5.7 Observation and renderer design

**Observation:** precompute static annotated collision channels per room at
layout-build time. `observe` gathers one `[H, W, 2]` entry and computes nearest
item/enemy from padded tables with validity/on-field masks. Preserve full legacy
observation by default; offer a compact observation only as an explicit new API.

**Renderer:** precompose each room's static background/decor into
`static_rgba[R, H, W, 4]`; gather it for the active room; overlay only padded
dynamic objects. Replace full-screen `render_at` stamping with a tested bounded
patch compositing strategy. Keep renderer state-free and apply the final HWC
conversion exactly once.

## 6. Detailed execution plan

Each phase has a deliverable and a hard exit gate. Do not begin a later phase by
deleting the legacy implementation before its reference tests exist.

### M0 — establish semantic baseline and decision ledger

**Inputs:** current baseline implementation and all shipped layouts.

**Work:**

1. Create a dedicated `tests/montezuma/` package and a legacy-reference fixture
   generator. Run each legacy layout in an isolated process because SANTAH is
   global.
2. Capture normalized deterministic traces, not whole legacy states. See section
   7 for the schema.
3. Make a small probe layout for every feature and capture short interaction
   traces that reach the feature reliably.
4. Write the decision ledger from section 4. Classify every observed known bug
   before implementation changes.
5. Fix test discovery so helper `jax_mzuma_*` modules are not treated as games.

**Deliverables:** versioned fixtures, probe action scripts, decision ledger, and
tests that execute against the old implementation.

**Exit gate:** baseline traces run deterministically on all supported layouts;
every intended deviation has an explicit test/decision.

### M1 — define new data contracts and layout compiler (host-only)

**Work:**

1. Add immutable schema classes for `LayoutCapacity`, `LayoutSpec`,
   `MontezumaLayout`, `SimState`, and component states. Annotate all axes/dtypes.
2. Implement a host-only `LayoutBuilder`/compiler. It must validate ids,
   capacity, bounds, reciprocal connections, image/mask shape, entity metadata,
   coordinate conversion, and opaque optional-mask usage.
3. Port the current one-room test layout first. Compile its room-local assets into
   full-screen static arrays and compare builder products to legacy projections.
4. Emit/load a versioned non-pickle layout artifact (`manifest` + fixed-shape
   arrays). Test byte-stable or canonical round trips, manifest validation,
   fingerprint changes on meaningful source changes, and one-time device
   placement at environment construction.
5. Ensure new layout objects have no SANTAH/global registration side effect.

**Suggested file boundary:**

```text
src/jaxatari/games/montezuma/schema.py
src/jaxatari/games/montezuma/layout_builder.py
src/jaxatari/games/montezuma/layout_artifact.py
src/jaxatari/games/montezuma/layouts.py
src/jaxatari/games/montezuma/assets.py
```

**Exit gate:** arbitrary generated layouts with the same capacity produce the
same tree structure; invalid specifications fail before JIT; one-room static
layout/observation assets match the reference; an artifact round trip binds the
same immutable bundle for simulation and rendering.

### M2 — implement pure core with direct dynamic-room state

**Work:**

1. Implement `reset_core`, action decoding, player/game scalar updates, and
   direct room-row gather/scatter with no rendering or observation caching.
   Then bind that core to a single `LayoutBundle` for the production reset/step
   functions; batch only `SimState` and action.
2. Port static room connections/spawn handling. A transition is an indexed
   lookup, not persistence write/read.
3. Port dynamic items, doors, enemies, timers, ladders/ropes, and reset behavior
   one feature family at a time. Update each selected row at most once per group.
4. Build semantic projection helpers so the new state can be compared with M0
   fixtures despite its intentionally different layout.

**Exit gate:** test-layout and a generated multi-room layout pass reset,
movement, transition, and persistence-across-visits semantic traces under
`jax.jit`, `lax.scan`, and a shared-layout `vmap`; no new core file imports
SANTAH/PyramidLayout.

### M3 — replace global collision composition and convolution

**Work:**

1. Implement a scalar/reference collision oracle for tests.
2. Implement local static/dynamic collision queries and bounded axis/sweep
   resolution.
3. Differential-test every legal player position/action on small synthetic
   rasters, then action traces in legacy rooms.
4. Remove `augmented_collision_map` from new simulation state only after parity.

**Exit gate:** no `jax.scipy.signal.convolve2d` in the new transition path; all
collision parity/fuzz tests pass; no tunnelling/boundary regression occurs.

### M4 — port all shipped layouts and procedural-authoring tests

**Work:**

1. Port demo, difficulty 1, difficulty 2, difficulty 3, combined layouts, and
   saved-state loading only if its semantics are intentionally retained.
2. Implement small generated layouts that exercise capacity padding, arbitrary
   connectivity, empty rooms, each feature, and differing room heights.
3. Check the compiler can choose capacity from a spec and can accept declared
   larger capacity without behavior changes.

**Exit gate:** every shipped layout has a compiled-layout test and M0 semantic
trace; generated-layout tests prove no hand-authored room branch is needed.

### M5 — observation refactor

**Work:**

1. Precompute static annotations in the compiler.
2. Implement `observe_bound(state)` without mutation/caching of state; retain
   an explicit-layout wrapper only for core-level tests.
3. Preserve fields, shape, dtype, flattening order, and coordinate semantics
   until a documented API change says otherwise.
4. Test full/compact modes separately if compact mode is added.

**Exit gate:** observation golden arrays and flattened wrapper observations pass;
normal step has no room dispatch and no full annotation composition beyond the
required output gather.

### M6 — renderer refactor

**Work:**

1. Make rendering a pure state-free function over the *same bound*
   `LayoutBundle` + `SimState`; do not independently load/build a renderer
   layout.
2. Precompose static layers; overlay padded dynamic feature/entity tables.
3. Replace each full-frame sprite stamp only after selected-frame image goldens
   pass. Keep freeze/death/dark-room layering explicit.
4. Benchmark raw render and Pixel/Atari wrapper paths independently.

**Exit gate:** exact selected-frame goldens across every visual feature; no
room-count dispatch; renderer has no second reconstructed PyramidLayout.

### M7 — façade, wrappers, and public integration

**Work:**

1. Make `JaxMontezuma(layout_artifact=..., config=...)` the production API and
   retain a documented default artifact constructor. Also expose a host-only
   `from_layout_spec(...)` path that compiles/binds (and may cache) an artifact.
2. Decide/register `jaxatari.make("montezuma", ...)` only after it can bind a
   programmatic layout/config without hidden globals or a framework-level
   batched-layout argument.
3. Standardize `lives` behavior and verify `AtariWrapper(episodic_life=True)`.
4. Test raw, Atari, object, pixel, flattened, logging, and Gym functional paths.
5. Define a versioned checkpoint manifest containing a layout fingerprint and
   schema version. Either provide a separately tested legacy-pickle migration
   tool or reject old pickles with a clear error; never unpickle them implicitly
   as the new runtime-state format.

**Exit gate:** all chosen public compatibility contracts pass; no wrapper relies
on legacy private state fields.

### M8 — performance validation and rollout

**Work:**

1. Run the benchmark matrix in section 8 on the target JAX/CUDA stack.
2. Compare old/new semantic outputs and performance for one-room, 5-room,
   24-room, 48-room, and 72-room layouts where available.
3. Run mixed-room-id and homogeneous-room-id batches with one shared bound
   layout. Compare the bound-artifact mode with the explicit unbatched-layout
   core as a diagnostic; inspect XLA/trace data rather than relying only on
   wall-clock time.
4. Keep the legacy engine only as a test/reference tool during migration; remove
   it and SANTAH coupling after all consumers have moved.

**Exit gate:** performance acceptance criteria in section 9 and all regression
tests pass on the supported stack.

## 7. Regression-test specification

### 7.1 Fixture format

Do not snapshot an entire internal state: its schema must change. Store compact
semantic records in NPZ/JSON metadata with a schema version, layout id, action
sequence, seed, and reference baseline hash.

```text
TraceRecord[t]
  room_id
  player_xy (legacy/public convention)
  movement_flags: standing, jumping, falling, climbing, laddering, on_rope
  score, lives, inventory, frame_count
  freeze: active, type, remaining
  nearest item/enemy projection
  dynamic records for active and previously touched rooms
  reward, done, info projection
  observation selected fields/hash
  optional RGB frame hash or exact selected frame
```

Use exact array comparisons for required deterministic fields. Use hashes only
for large images when a failing test also writes a readable diff artifact.

### 7.2 Required test modules

```text
tests/montezuma/
  test_layout_compiler.py
  test_layout_artifact.py
  test_layout_generation.py
  test_semantic_regression.py
  test_mechanics.py
  test_collision.py
  test_observation.py
  test_render_regression.py
  test_jax_transforms.py
  test_wrapper_contract.py
  fixtures/v1/...
```

### 7.3 Mechanics matrix

| Area | Required assertions |
| --- | --- |
| Reset/state | initial room, pose, inventory, score/lives, RNG determinism, every state leaf shape/dtype |
| Layout artifact | manifest/schema/fingerprint validation, host-only load, no artifact leaves in `SimState`, one bundle shared by game and renderer |
| Input/motion | horizontal boundaries, idle, action mapping, jump hold, jump arc, fall, one-way/platform behavior |
| Collision | player hitbox exactness, local sweep parity, boundary solidity, no tunnelling, down-ray/fall behavior |
| Rooms | each exit direction, no-edge behavior, reciprocal connection, entry spawn, ladder-aware entry, persistence across revisits |
| Items | pickup, inventory limits, gem exception, hammer timer, sword/key consumption, bonus reposition RNG |
| Doors | collision, key gate, unlock state persists, freeze/reward behavior |
| Enemies | snake/rolling/bounce/spider motion, animation, reset-on-room-entry, sword/hammer outcomes, split bounce skull |
| Environment features | ladders, ropes, conveyors, lasers and timing, dropout floors, sidewalls, pit, dark room, bonus cycle/reset |
| Death/freeze | each freeze type, rendered pre/post-step semantics, resume, life loss, terminal behavior |
| Observation | annotation channels, nearest entities, flags, flatten order/space ranges, legacy coordinate contract |
| Rendering | static room, each feature layer, player modes, freeze/death frames, score/lives/inventory, HWC uint8 output |
| Wrappers | raw, Atari frame skip/stack, object, pixel, combined/flattened/logging, episodic life, FuncEnv |
| Checkpoints | layout fingerprint/schema-version validation; explicit migration or deliberate rejection of legacy pickles |

### 7.4 Transform and property tests

At minimum assert:

```text
jit(reset), jit(step), and lax.scan(step) succeed
vmap(reset/step) succeeds with one shared bound LayoutBundle
vmap(step) succeeds when lanes intentionally occupy different room ids in that layout
all state/output leaves retain shape and dtype across room changes
same seed + actions gives identical trace
different same-capacity generated layouts have the same functional-core signature
artifact round-trip/load binds one immutable bundle without adding it to state
padding entries cannot affect collisions, observations, rendering, or rewards
```

Use synthetic tiny/probe layouts for exhaustive collision and connection tests;
use real layouts for integration traces. Do not skip mixed-room vmap because of
the old implementation's memory behavior—the rewrite specifically exists to
support it.

## 8. Benchmark protocol

Create a dedicated benchmark, for example
`scripts/benchmarks/montezuma_performance.py`; do not modify the existing
Pong-oriented wrapper benchmark to infer Montezuma behavior.

### Environment metadata captured with every result

```text
git revision, Python, jax, jaxlib, CUDA/driver, device name/count,
XLA flags, layout fingerprint/capacity, artifact schema and delivery mode,
render mode, batch size, action distribution, warmup count, timed steps,
median/p50/p95, artifact load/device-place time, compile time, peak device memory
```

### Benchmark matrix

| Dimension | Values |
| --- | --- |
| Layout size | 1-room test, 5-room demo, 24-room difficulty, 48-room, 72-room, generated stress layouts |
| Layout delivery | production bound artifact/shared bundle; explicit unbatched layout parameter (diagnostic only) |
| API path | pure transition only; public raw step; object wrapper; render only; Pixel/Atari wrapper |
| Batch | 1, 32, 256, 1024, and target training batch sizes |
| Room distribution | all lanes same room; evenly mixed rooms; adversarially different feature-rich rooms |
| Actions | no-op; same action; seeded random actions; directed transition trace |
| Measurement | first compile/reset; first step compile; warmed scan throughput; memory; trace/HLO size |

### Measurement rules

1. Use the project target JAX/CUDA version, not the local CPU diagnostic
   environment, for release numbers.
2. Warm JIT explicitly and call `block_until_ready()` outside the timed loop.
3. Use `lax.scan` for a long on-device roll-out so Python dispatch does not
   dominate. Report reset/compile separately.
4. Profile a warmed window with JAX profiler/Chrome trace and, where possible,
   Nsight. Record device utilization and kernel count.
5. Inspect lowered StableHLO/HLO size or op counts across increasing room count.
   It is a structural diagnostic, not the only benchmark.
6. Keep collision microbenchmarks (legacy convolution vs new local query) and
   renderer microbenchmarks separate from end-to-end numbers.
7. Verify layout memory is not multiplied by batch size: report static bundle
   bytes once, `SimState` bytes per environment, and device-memory growth as B
   increases. Time artifact load/device placement separately from warmed steps.

## 9. Performance acceptance criteria

Set numerical throughput targets after recording the target-device baseline, but
the following structural gates are non-negotiable:

1. Hot simulation/observation/render paths have no `lax.switch` whose branch
   count depends on `num_rooms`.
2. New transition code has no SANTAH raise/lower/serialize/load operation and
   no full-playfield `convolve2d` for player correction.
3. Lowered core graph/operator count is approximately stable as only `R` grows
   with all per-room capacities fixed; it must not multiply by the number of
   room-aware handlers.
4. Same-layout mixed-room vmap throughput must be measured and must not suffer
   the legacy branch explosion. Set a recorded ratio target relative to the
   homogeneous batch after baseline data exists.
5. State memory excludes redundant canvas, augmented collision, and observation
   leaves **and all static LayoutBundle leaves**. Report per-environment state
   bytes, one shared layout-bundle byte count, and batch peak memory.
6. Rendering has a separate budget and must not invalidate simulation gains.
7. Game and renderer bind the same validated layout fingerprint; no renderer
   construction may rebuild or duplicate static layout tables per environment.
8. All performance claims include compile, warmed runtime, layout capacity,
   delivery mode, and API path; never report a single ambiguous FPS number.

## 10. Migration safety rules

- Work on a new engine/module beside the legacy one until M0–M5 pass. Avoid a
  wholesale in-place rewrite that destroys the only behavioral oracle.
- Port one feature family at a time and keep semantic projections comparable.
- Make padding/masks visible in the schema and builder validations, never as
  hidden defaults in gameplay rules.
- Prefer a single authoritative compiler for static maps, observations, and
  render layers. Duplicated game/renderer layout construction is forbidden.
- Keep host-only layout generation outside jitted functions. Runtime code gets
  arrays, not Python lists/dicts/types; artifact I/O is construction-time only.
- Bind one immutable layout artifact per production environment/batch. Never
  place it in `SimState` or a batched wrapper state; change it only between
  runs by constructing/binding a new environment.
- Treat any state-schema change as intentional. Wrapper compatibility is tested,
  not assumed.
- Once migration is complete, remove obsolete SANTAH/proto/persistence code
  rather than keeping two runtime architectures indefinitely.

## 11. Handoff checklist

Before an agent starts implementation, it should be able to answer yes to all:

- [ ] I know whether I am changing layout compilation, simulation, observation,
  rendering, or façade/wrapper code.
- [ ] I have read the M0 fixture/decision status and know the expected semantic
  result for my feature.
- [ ] My new runtime arrays have explicit axes, capacity, mask, and dtype.
- [ ] I know whether this code consumes the single bound `LayoutBundle` or the
  optional explicit functional-core layout, and I did not add layout data to
  `SimState` or batch axes.
- [ ] If I touch loading/rendering, I validate the artifact fingerprint and
  share the exact bound bundle instead of rebuilding layout data.
- [ ] My implementation uses gather/scatter or masked vectorization, not a
  per-room Python/`lax.switch` dispatch.
- [ ] I added a targeted mechanics test plus a JIT/vmap shape test where
  applicable.
- [ ] I ran the relevant raw and wrapper-level regression tests.
- [ ] If performance-sensitive, I measured compile and warmed device execution
  separately and recorded the layout/batch distribution.

## 12. Reconnaissance provenance

This document was produced without modifying Montezuma runtime code. It is
grounded in baseline source paths cited above, direct host-side shape inspection
of the difficulty-1 layout, and directional CPU-only diagnostics. Re-run the
benchmark protocol on the target GPU/JAX stack before making quantitative claims
or locking thresholds.
