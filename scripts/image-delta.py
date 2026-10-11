#!/usr/bin/env python3
"""Shrink a `docker save` archive to the layers the receiver lacks.

`docker load` keeps a layer it already has and never opens that layer's file
in the archive, so those files can be left out. An 11 GB Jetson image whose
base is already on the board then travels as a few megabytes.

  image-delta.py id ARCHIVE
      print the image ID (digest of the image config) in the archive
  image-delta.py strip ARCHIVE DIFF_ID... > smaller.tar
      write the archive without the layers whose diff IDs are listed
"""
import json
import sys
import tarfile


def read_json(archive, name):
    return json.load(archive.extractfile(name))


def main():
    mode, path, have = sys.argv[1], sys.argv[2], set(sys.argv[3:])
    with tarfile.open(path) as archive:
        manifest = read_json(archive, "manifest.json")[0]
        if mode == "id":
            # "blobs/sha256/<digest>"
            print("sha256:" + manifest["Config"].rsplit("/", 1)[-1])
            return

        # Layer files are listed in the same order as the config's diff IDs
        diff_ids = read_json(archive, manifest["Config"])["rootfs"]["diff_ids"]
        skip = {
            layer
            for layer, diff_id in zip(manifest["Layers"], diff_ids)
            if diff_id in have
        }
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as out:
            for member in archive:
                if member.name in skip:
                    continue
                out.addfile(
                    member, archive.extractfile(member) if member.isfile() else None
                )
        print(
            f"kept {len(diff_ids) - len(skip)} of {len(diff_ids)} layers",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
