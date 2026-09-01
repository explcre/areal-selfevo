#!/usr/bin/env bash
# Configure podman to work INSIDE an unprivileged container, then verify it the way harbor
# will actually use it.
#
# Rootless podman defaults to a kernel overlayfs mount, which an unprivileged container is not
# allowed to perform:
#     Error: mount /var/lib/containers/storage/overlay: permission denied
# The fix is a storage driver that needs no privileged mount -- fuse-overlayfs where /dev/fuse
# is available, otherwise vfs, which always works but copies whole layers and is slow.
#
#   bash setup_podman.sh            # configure, then verify
#   FORCE_VFS=1 bash setup_podman.sh  # skip fuse-overlayfs and go straight to vfs
#
# Safe to re-run. Backs up any existing storage.conf before writing.
set -u

CONF_DIR=/etc/containers
CONF="$CONF_DIR/storage.conf"
RUNROOT="${PODMAN_RUNROOT:-/run/containers/storage}"
GRAPHROOT="${PODMAN_GRAPHROOT:-/var/lib/containers/storage}"

ok(){   printf '  \033[32mok\033[0m    %s\n' "$1"; }
bad(){  printf '  \033[31mFAIL\033[0m  %s\n' "$1"; [ $# -gt 1 ] && printf '        -> %s\n' "$2"; }
warn(){ printf '  \033[33mwarn\033[0m  %s\n' "$1"; }

command -v podman >/dev/null 2>&1 || {
  bad "podman not installed" "apt-get install -y podman podman-docker podman-compose"
  exit 2
}

for d in "$CONF_DIR" "$RUNROOT" "$GRAPHROOT"; do
  [ -d "$d" ] && continue
  mkdir -p "$d" 2>/dev/null || {
    if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then sudo mkdir -p "$d" 2>/dev/null; fi
  }
  [ -d "$d" ] || warn "could not create $d; podman may refuse to start"
done
# Quiets podman-docker's "Emulate Docker CLI using podman" banner, which otherwise contaminates
# the stdout of every command harbor parses.
touch "$CONF_DIR/nodocker" 2>/dev/null

# Pick the driver from what the kernel will actually allow, not from what we hope.
DRIVER="vfs"; MOUNT_PROG=""
if [ "${FORCE_VFS:-0}" != "1" ] && [ -c /dev/fuse ] && command -v fuse-overlayfs >/dev/null 2>&1; then
  DRIVER="overlay"; MOUNT_PROG="$(command -v fuse-overlayfs)"
  ok "using fuse-overlayfs ($MOUNT_PROG); /dev/fuse present"
else
  [ -c /dev/fuse ] || warn "/dev/fuse absent, so fuse-overlayfs cannot be used"
  warn "falling back to the vfs driver: always works, but copies whole layers and is slow"
fi

[ -f "$CONF" ] && cp -a "$CONF" "$CONF.bak.$(date +%s)" && ok "backed up existing $CONF"

# runroot and graphroot are REQUIRED once a [storage] section exists; omitting them fails with
# "runroot must be set", which reads like a podman bug rather than a missing key.
# Written to a temp file first so a permission failure is caught rather than reported as
# success. An earlier version printed "wrote ..." unconditionally, and on a box without root
# it claimed to have configured storage it had not touched.
TMPCONF="$(mktemp 2>/dev/null || echo /tmp/storage.conf.$$)"
{
  echo '[storage]'
  echo "driver = \"$DRIVER\""
  echo "runroot = \"$RUNROOT\""
  echo "graphroot = \"$GRAPHROOT\""
  if [ -n "$MOUNT_PROG" ]; then
    echo ''
    echo '[storage.options.overlay]'
    echo "mount_program = \"$MOUNT_PROG\""
  fi
} > "$TMPCONF" 2>/dev/null
if [ ! -s "$TMPCONF" ]; then
  bad "could not compose a storage config" "Could not write to $(dirname "$TMPCONF")."
  exit 3
fi
if cp "$TMPCONF" "$CONF" 2>/dev/null; then
  rm -f "$TMPCONF"
  ok "wrote $CONF (driver=$DRIVER)"
elif [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && sudo cp "$TMPCONF" "$CONF" 2>/dev/null; then
  rm -f "$TMPCONF"
  ok "wrote $CONF via sudo (driver=$DRIVER)"
else
  bad "cannot write $CONF" "Need root. Re-run as root, or copy it yourself:
             sudo cp $TMPCONF $CONF
           Config left at $TMPCONF so nothing is lost."
  exit 3
fi

echo "== verify =="
FAILED=0
if podman info >/dev/null 2>&1; then
  ok "podman info"
else
  bad "podman info still fails" "Full error below. If it still names an overlay mount, re-run as: FORCE_VFS=1 bash $0"
  podman info 2>&1 | sed 's/^/        /' | tail -12
  FAILED=1
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  ok "docker shim works"
else
  bad "docker shim not working" "apt-get install -y podman-docker"
  FAILED=1
fi

# harbor runs every task through `docker compose`, never plain `docker run`, so this is the
# check that decides whether Terminal-Bench can work here at all.
if docker compose version >/dev/null 2>&1; then
  ok "docker compose: $(docker compose version 2>/dev/null | head -1)"
else
  bad "'docker compose' unavailable" "harbor drives EVERY task through it. Try: apt-get install -y podman-compose
           (podman then delegates 'podman compose' to it, and the docker shim maps onto that)"
  FAILED=1
fi

# Pulling an image is the step most likely to fail behind an intercepting proxy, and it would
# otherwise fail per-task, long after setup looked successful.
if timeout 300 docker run --rm hello-world >/dev/null 2>&1; then
  ok "container run + registry pull"
else
  bad "cannot run a container pulled from the registry" \
      "Run 'docker pull hello-world' and read the error. A TLS error means the proxy intercepts
           registry traffic; a permission error means the storage driver still is not usable."
  FAILED=1
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "podman is usable; next: bash experiments/harness/run_tb_swap.sh --smoke"
else
  echo "podman is NOT usable for Terminal-Bench on this box."
  echo "That is a legitimate answer, not a failure to report. The GPU scoring path needs no"
  echo "containers at all:  bash experiments/harness/run_h200_math.sh --install"
fi
exit "$FAILED"
