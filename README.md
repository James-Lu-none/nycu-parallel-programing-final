# parallel programming implementation for Three‑Body Problem Simulation

## build and run

```bash
mkdir build && cd build
# ACCEL_VARIANT can be one of: 
# serial, serial_simd
# pthread_blocked, pthread_interleaved, pthread_mutex_blocked, pthread_mutex_interleaved, 
# pthread_simd_blocked, pthread_simd_interleaved, pthread_mutex_simd_blocked, pthread_mutex_simd_interleaved
# cuda_blocked, cuda_interleaved
# 
# RENDER_VARIANT can be one of:
# serial, serial_simd
# pthread, pthread_simd, pthread_mutex, pthread_mutex_simd
# cuda

# full serial variant
cmake -DACCEL_VARIANT=serial -DRENDER_VARIANT=serial ..
# fastest cpu variant
cmake -DACCEL_VARIANT=pthread_mutex_interleaved -DRENDER_VARIANT=pthread_mutex_simd ..
# fastest combined cpu + gpu variant
cmake -DACCEL_VARIANT=pthread_mutex_simd_interleaved -DRENDER_VARIANT=cuda ..

make

# Run N-body simulation with input from file
./N_body ../assets/clustered_1000.txt 8
```

## build tracy tools

```bash
git clone https://github.com/wolfpld/tracy
git checkout v0.12.2
cd tracy

cmake -B ./capture/build -S ./capture -DCMAKE_BUILD_TYPE=Release
cmake --build ./capture/build --parallel

cmake -B ./csvexport/build -S ./csvexport -DCMAKE_BUILD_TYPE=Release
cmake --build ./csvexport/build --parallel
```

## accelerations_thread_v1

In accelerations_thread_v1, the original iteration on NUM_BODIES is divided to NUM_THREADS to perform parallel acceleration calculation,
but in the calculation, every acceleration component is calculated in pair by exploiting the Newton’s 3rd law of motion
(for body number i, acceleration calculations are only applied from body number i+1 to the last body),
so the load in each thread becomes imbalanced, we can derive the calculation count as follow:

let P = NUM_BODIES/NUM_THREADS where P>=1
calc_count for thread i
= sum(NUM_BODIES-k-1, k, P*i, P*(i+1))
= (P*(NUM_THREADS-i-1)+P*(NUM_THREADS-i)-1)*P/2

And here is the runtime result:
NUM_BODIES = 100
NUM_THREADS = 4

```log
Thread 3: i_start=75, i_end=100, calc_count=300
Thread 2: i_start=50, i_end=75, calc_count=925
Thread 0: i_start=0, i_end=25, calc_count=2175
Thread 1: i_start=25, i_end=50, calc_count=1550
Thread 3: i_start=75, i_end=100, calc_count=300
Thread 2: i_start=50, i_end=75, calc_count=925
Thread 1: i_start=25, i_end=50, calc_count=1550
Thread 0: i_start=0, i_end=25, calc_count=2175
Thread 3: i_start=75, i_end=100, calc_count=300
Thread 2: i_start=50, i_end=75, calc_count=925
Thread 0: i_start=0, i_end=25, calc_count=2175
Thread 1: i_start=25, i_end=50, calc_count=1550
```

## accelerations_thread_v2

In accelerations_thread_v2, the original iteration on NUM_BODIES is divided to NUM_THREADS to perform parallel acceleration calculation,
but different from accelerations_thread_v1, the acceleration component is calculated independently for each body i,
although this approach adds redundant calculation in total, the load on each thread is balanced, we can derive the calculation count as follow:

let P = NUM_BODIES/NUM_THREADS where P>=1
calc_count for thread i
= P*NUM_BODIES

And here is the runtime result:
NUM_BODIES = 100
NUM_THREADS = 4

```log
Thread 0: i_start=0, i_end=25, calc_count=2500
Thread 1: i_start=25, i_end=50, calc_count=2500
Thread 2: i_start=50, i_end=75, calc_count=2500
Thread 3: i_start=75, i_end=100, calc_count=2500
Thread 0: i_start=0, i_end=25, calc_count=2500
Thread 1: i_start=25, i_end=50, calc_count=2500
Thread 2: i_start=50, i_end=75, calc_count=2500
Thread 3: i_start=75, i_end=100, calc_count=2500
Thread 0: i_start=0, i_end=25, calc_count=2500
Thread 1: i_start=25, i_end=50, calc_count=2500
Thread 2: i_start=50, i_end=75, calc_count=2500
Thread 3: i_start=75, i_end=100, calc_count=2500
```

## error: malloc(): unaligned fastbin chunk detected

```
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
Thread 2 resetting hasWork flag and starting work
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
***** thread 2 is done! *****
===== all threads are done! merging result =====
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
Thread 2 resetting hasWork flag and starting work
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
***** thread 2 is done! *****
===== all threads are done! merging result =====
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
Thread 2 resetting hasWork flag and starting work
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
***** thread 2 is done! *****
===== all threads are done! merging result =====
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
***** thread 2 is done! *****
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
Thread 2 resetting hasWork flag and starting work
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
***** thread 0 is done! *****
***** thread 1 is done! *****
***** thread 2 is done! *****
===== all threads are done! merging result =====
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
Thread 2 resetting hasWork flag and starting work
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
***** thread 2 is done! *****
===== all threads are done! merging result =====
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
====== Wake up all workers =====
***** thread 0 is done! *****
***** thread 1 is done! *****
Thread 2 resetting hasWork flag and starting work
Thread 1 resetting hasWork flag and starting work
Thread 1 finished work and is signaling completion to main thread
Thread 1 entered critical session
Thread 1 has no work and waiting to be signaled
Thread 2 finished work and is signaling completion to main thread
Thread 2 entered critical session
Thread 2 has no work and waiting to be signaled
Thread 0 resetting hasWork flag and starting work
Thread 0 finished work and is signaling completion to main thread
Thread 0 entered critical session
Thread 0 has no work and waiting to be signaled
***** thread 2 is done! *****
===== all threads are done! merging result =====
```

## Three‑Body Problem Simulation

A real‑time visualisation of the famous planar **figure‑8 three‑body orbit**. The program integrates Newtonian gravity with a **leapfrog (velocity‑Verlet) scheme**, recenters the camera on the instantaneous centre of mass, and leaves colourful trails behind each body.

---

## Demo

![Demo](../assets/3_body_simulation.gif)

## 1. Quick Start

### Build

```bash
chmod +x build.sh
./build
```

> Requires the SDL2 development package (`libsdl2-dev` on Debian/Ubuntu, `brew install sdl2` on macOS).

### Run

```bash
./3_body
```

A window **1200 × 800** opens; three coloured blobs chase each other in a repeating figure‑8. Close the window or press the close button to quit.

---

## 2. Key Features

| Feature                         | Detail                                                                                       |
| ------------------------------- | -------------------------------------------------------------------------------------------- |
| **Order‑4 Leapfrog Integrator** | Time‑reversible, symplectic; good energy behaviour.                                          |
| **Figure‑8 Initial Condition**  | Uses the well‑known equal‑mass periodic orbit (Chenciner & Montgomery, 2000).                |
| **Adaptive Frame Timing**       | Physics time‑step `FIXED_DT = 2 × 10⁻⁴ s`; real‑time frames accumulate leftover time.        |
| **Centre‑of‑Mass Re‑centering** | Keeps the dance centred in the window regardless of momentum drift.                          |
| **Trail Buffer per Body**       | Circular buffer (`TRAIL_BUF = 5000`) with min‑distance filter to avoid over‑plotting points. |
| **Pure SDL Surface Drawing**    | No hardware renderer, just pixel‑fills; portable and simple.                                 |

---

## 3. Code Tour

### 3.1 Configuration Macros

```c
#define WIDTH  1200
#define HEIGHT  800
#define NUM_BODIES 3
#define G 10000.0          // Gravitational constant (scaled for simulation)
#define FIXED_DT 0.0002    // Physics step (seconds)
```

* Screen size can be changed; physics scale is arbitrary (pixels ≈ distance units).
* `G` is tuned so the figure‑8 fits nicely inside the window.

### 3.2 `Planet` & `Trail` Structs

```c
typedef struct {
    float x, y;      // position (px)
    float vx, vy;    // velocity (px/s)
    float mass;      // mass (arbitrary units)
    float r;         // radius when drawn (px)
} Planet;

typedef struct {
    int x[TRAIL_BUF]; // circular arrays of pixel coords
    int y[TRAIL_BUF];
    int head;         // next write position
    int size;         // current valid points
} Trail;
```

* Trails store integers to keep memory small and drawing fast.
* `MIN_DIST = 1.5` px avoids redundant successive points.

### 3.3 Drawing Helpers

* **`fill_circle`**: brute‑force pixel fill inside `r² = x²+y²` (good enough for small radii).
* **`trail_push`**: append point if moved ≥ `MIN_DIST` from last.
* **`trail_draw`**: draws from newest to oldest (optional; ordering doesn’t matter visually).

### 3.4 Physics – Leapfrog Step

```c
// 1. Half‑kick
vx += 0.5·a·dt
// 2. Drift
x  += vx·dt
// 3. Recompute accelerations (new positions)
// 4. Another half‑kick
vx += 0.5·a·dt
```

* `accelerations()` computes pairwise forces with softened denominator (`+ EPSILON`) to avoid singularities.
* Arrays `ax[]`, `ay[]` are static to avoid realloc each frame.

### 3.5 Camera Recentering

```c
cx = Σ mᵢ xᵢ / Σ mᵢ;
cy = Σ mᵢ yᵢ / Σ mᵢ;
dx = WIDTH/2 - cx;  // shift so CoM at centre
for each body: xᵢ += dx, yᵢ += dy;
```

Keeps the system onscreen even if numerical drift introduces net momentum.

---

## 4. Controls & Interaction

There are no interactive controls in this minimal build—close the window to exit. You can easily add:

| Key     | Action                             |
| ------- | ---------------------------------- |
| **P**   | Pause/unpause integration          |
| **R**   | Reset to initial condition         |
| **↑/↓** | Increase/decrease simulation speed |

---

## 5. Performance Notes

* **CPU only**: Each frame touches \~`NUM_BODIES²` force pairs (here 3 → 3 pairs). Even with hundreds of bodies, leapfrog + simple force loop is fine.
* **Trail cost**: drawing `TRAIL_BUF × NUM_BODIES` 2×2 px rects each frame; negligible.
* Main cost is **full‑surface clear** (`SDL_FillRect`) each frame. Switch to SDL renderer/textures for hardware acceleration if needed.