# T1 -- sentaurus-fefet, containerised.
#
# This image contains NO Synopsys code. It copies the driver and nothing else,
# and expects the Sentaurus installation to be bind-mounted at /opt/synopsys by
# whoever holds the licence:
#
#   docker build -t sentaurus-fefet:byol -f t1/Dockerfile .
#   docker run --rm \
#     -v /path/to/Sentaurus/V-2023.12:/opt/synopsys:ro \
#     -e SNPSLMD_LICENSE_FILE=27020@your-license-server \
#     -e ORACLE_BACKEND=sentaurus \
#     sentaurus-fefet:byol serve
#
# Redistributing the solver would violate the licence; redistributing a recipe
# for wiring it up does not. See docs/T1_CONTAINER.md for why the flagship run
# itself is uncontainerised.

FROM python:3.12-slim

# csh, because ~/.cshrc uses setenv and `set path` and bash cannot source it, and
# the PuTTY tools for the remote path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tcsh \
        putty-tools \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tesseract

COPY pyproject.toml README.md ./
COPY src/diffsilicon ./diffsilicon
COPY config ./config
COPY tesseracts/sentaurus-fefet/tesseract_api.py ./tesseract_api.py
COPY t1 ./t1

RUN pip install --no-cache-dir \
        "tesseract-core[runtime]>=1.11.0" \
        "jax[cpu]>=0.4.35" \
        "numpy>=1.26" \
        "pyyaml>=6.0"

ENV TESSERACT_API_PATH=/tesseract/tesseract_api.py \
    DIFFSILICON_CONFIG=/tesseract/config/circuit.yaml \
    ORACLE_BACKEND=sentaurus \
    PLINK=/usr/bin/plink \
    PSCP=/usr/bin/pscp \
    PYTHONUNBUFFERED=1

# Sentaurus lands here at run time. Empty in the image, by licence and on purpose.
VOLUME ["/opt/synopsys"]

ENTRYPOINT ["tesseract-runtime"]
CMD ["--help"]
