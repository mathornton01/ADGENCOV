# ADGENCOV service — portable production image for Railway / any container host.
#
# Two stages:
#   1. builder  — compiles the C++/pybind11 core (adgencov._core) with a
#                 PORTABLE instruction set (-march=native OFF) so the resulting
#                 binary runs on whatever CPU the platform schedules us onto,
#                 not just the build host.  Eigen + pybind11 are fetched at
#                 configure time (needs git + network, both available in build).
#   2. runtime  — a slim image with only the Python runtime deps (API + GEO)
#                 and the built package.  No compiler, no Eigen, no build cache.
#
# The FastAPI app binds to $PORT (Railway injects it; defaults to 8000 locally).

# ---- Stage 1: build the native extension --------------------------------
FROM python:3.12-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Portable build: CLI/tests off, Python on, native SIMD OFF for portability.
# The parallel-grid speedup lives in Python and is unaffected by this; only the
# per-op SIMD width is traded away for a binary that runs on any x86-64 host.
RUN cmake -S . -B build-docker \
        -DCMAKE_BUILD_TYPE=Release \
        -DADGENCOV_BUILD_PYTHON=ON \
        -DADGENCOV_BUILD_CLI=OFF \
        -DADGENCOV_BUILD_TESTS=OFF \
        -DADGENCOV_NATIVE_ARCH=OFF \
    && cmake --build build-docker --target adgencov_core -j "$(nproc)"

# Sanity: the compiled module must have landed next to the Python sources.
RUN test -f python/adgencov/_core*.so

# ---- Stage 2: slim runtime ----------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

# libstdc++6 / libgomp1: runtime C++ libs the compiled .so links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring over the Python package WITH the freshly built _core*.so inside it.
COPY --from=builder /src/python /app/python

# Install the package plus its API + GEO runtime extras. The .so is packaged
# via package-data (*.so) so no recompilation happens here.
RUN pip install --no-cache-dir "./python[api,geo]"

ENV PYTHONUNBUFFERED=1

# Documented default; Railway overrides via $PORT.
EXPOSE 8000

# Shell form so ${PORT} is expanded at runtime (Railway sets it; 8000 locally).
CMD ["sh", "-c", "uvicorn adgencov.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
