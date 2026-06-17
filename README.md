# RayTracer – Custom CPU Ray Tracer with Rasterization Preview

A full-featured CPU-based ray tracer written in C++17, complemented by a Z-buffer rasterization preview mode. The renderer leverages OpenMP parallelism, SIMD vector math (SSE/AVX), and multiple spatial acceleration structures to achieve interactive performance on ordinary multi-core CPUs.

## Features

- **Ray Tracing Core**  
  Implements a path tracing loop with direct illumination, shadows, specular reflections, refractions, and Fresnel effects. Uses iterative accumulation with Russian roulette termination to balance quality and performance.

- **Three Acceleration Structures**  
  - Linear scanning  
  - Bounding Volume Hierarchy (BVH)  
  - Uniform grid  
  Switch between them in real time to compare their performance for different scene compositions.

- **Geometry Support**  
  Spheres, triangles, planes, and triangle meshes loaded from OBJ files. All geometries participate in both ray tracing and rasterized preview.

- **Material System**  
  - Lambertian (diffuse)  
  - Metal (with fuzziness/roughness control)  
  - Dielectric (glass-like, with absorption and Fresnel effect)

- **Light Sources**  
  Point lights (with attenuation) and directional lights. Shadows are generated for all lights and can be toggled on/off for performance tuning.

- **Interactive Real‑time Rendering**  
  - WASD for movement, Space/Shift for vertical motion  
  - IJKL for view rotation, QE for roll  
  - Y to reset the up vector  
  - T to toggle auto‑orbit trajectory  
  - Z to instantly switch between ray tracing and rasterization preview

- **Rasterization Preview**  
  A CPU‑based Z‑buffer rasterizer that triangulates the scene on the fly. It provides fast feedback for scene layout and camera navigation, showing approximate diffuse colours.

- **SIMD‑Accelerated Math**  
  Vectors, matrices, and quaternions use SSE instructions for fast dot products, cross products, normalization, and transformations.

- **Multi‑threading**  
  Uses OpenMP to parallelize pixel processing (ray tracing) or triangle processing (rasterization), fully utilising available CPU cores.

- **Command‑Line Configuration**  
  Supports settings for resolution, object count, samples per pixel, recursion depth, shadow toggle, thread count, random seed, and output directories – ideal for automated testing and batch rendering.

- **Off‑line Rendering**  
  - Single‑frame BMP export  
  - Multi‑frame sequence output (along an orbit) with automatic ffmpeg video assembly (ffmpeg must be installed)

- **Live Performance Monitor**  
  The window title displays real‑time object count, seconds per frame, active acceleration method, and resolution – making performance evaluation straightforward.

---

## Project Structure

The code is organised into clearly separated modules: math library, geometry primitives, acceleration structures, scene management, camera, materials, and utilities – allowing easy extension and maintenance.