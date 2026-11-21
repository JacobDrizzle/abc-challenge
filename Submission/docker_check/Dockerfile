# A "clean Linux machine" for the submission, as the Submitting Artifacts page describes it:
# fresh OS, then only `pip install -r requirements.txt`, then run_inference.py with no network.
#
# Build context = the submission folder itself (or the unzipped archive), so nothing else from the repo gets in.
# The check script comes from a second, named build context ("checks") for the same reason.
# run_checks.sh does the build and run for several base images; README.md shows the single-image commands.

ARG BASE=python:3.12-slim
FROM ${BASE}

ARG SUBMISSION_NAME=Motion_Pending_headpose
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CUDA base images ship without Python: add the distro's own (Ubuntu 24.04 gives 3.12). python:*-slim skips this.
RUN if ! command -v python3 >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends python3 python3-venv && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# An empty virtual environment, so no package can come from the base image.
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Exactly what is in the archive folder.
COPY . /opt/${SUBMISSION_NAME}
WORKDIR /opt/${SUBMISSION_NAME}

# The only install step the organisers allow. If a pinned wheel doesn't exist for this Python, the build stops here.
RUN pip install -r requirements.txt

COPY --from=checks check_submission.py /checks/check_submission.py
CMD ["python", "/checks/check_submission.py"]
