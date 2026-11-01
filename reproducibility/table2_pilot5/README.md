# Archived five-object ablation source

This archive preserves the assemblies pilot: seven 2D conditions on five objects. It did not run the ablated SegViGen backends and is not the later full-cohort experiment.

The original `source_hashes.json` is copied from the pilot result bundle. All eleven Python files retain their bytes; Git line-ending conversion is disabled for this source directory. The release tests verify hashes without executing the archive.

`source/run_variant.py` deliberately requires five unique UIDs, checks RAW/source hashes, and reuses full-method planning caches. Variants are ours, box_point, single_positive, no_negative, no_feedback, and no_point_fusion; automatic SAM2 is a reused baseline.

Manifests, images, masks, references, palette, weights, planning caches, and environments are not included. Archived launch/preparation scripts contain historical paths: start_job.py selects a GPU and starts work; prepare_server.py writes a run directory. They are provenance records, not fresh-checkout commands. Do not remove fixed-cohort or hash guards to imply full-dataset reproduction.

The maintained V5 driver has later annotation-export changes and a different hash. Use this archive to inspect the original pilot behavior.
