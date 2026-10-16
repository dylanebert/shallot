FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive BUN_INSTALL=/opt/bun
ENV PATH=/opt/bun/bin:/opt/node/bin:$PATH
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
ENV PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright CI=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl xz-utils unzip git mesa-vulkan-drivers libvulkan1 libatomic1 \
    && rm -rf /var/lib/apt/lists/*
COPY .bun-version .node-version /tmp/pins/
RUN curl -fsSL https://bun.sh/install | bash -s "bun-v$(cat /tmp/pins/.bun-version)" \
    && arch=$(dpkg --print-architecture) && if [ "$arch" = amd64 ]; then arch=x64; fi \
    && version=$(cat /tmp/pins/.node-version) \
    && curl -fsSL "https://nodejs.org/dist/v$version/node-v$version-linux-$arch.tar.xz" -o /tmp/node.tar.xz \
    && mkdir /opt/node \
    && tar -xJf /tmp/node.tar.xz --strip-components=1 -C /opt/node \
    && rm /tmp/node.tar.xz
COPY package.json bun.lock /tmp/deps/
RUN cd /tmp/deps && bun install --frozen-lockfile --ignore-scripts \
    && bunx playwright install-deps chromium \
    && rm -rf /tmp/deps /root/.bun/install/cache /var/lib/apt/lists/*
WORKDIR /work
