Encode new video clips into the SANA-WM latent cache.

The loader pairs `<data_dir>/<shard>.zip` (mp4 + json entries) with
`<vae_cache_dir>/<shard>.zip` (`<key>.npz` holding `z` of shape (C,T,H,W)).
A shard is unusable unless both exist and the keys match.

1. Land raw clips as `<key>.mp4` + `<key>.json` inside one zip. Keep clips at the
   corpus length (961 frames @ 16 fps) unless you also change `data.num_frames`.
2. Encode with the LTX-2 VAE at 704x1280, the same one training loads:
   `vae_pretrained` points at the local stage-1 bundle, `vae_stride: [8, 32, 32]`.
   Latents are ~22 MB per 961-frame clip.
3. Write camera poses to `<shard>_camera.npz` keyed by the same clip keys. Without
   poses the camera branch gets no supervision and navigation metrics will not move.
4. Write at least one caption sidecar `<shard>_<suffix>.json` and reference the suffix
   from `external_caption_suffixes` + `caption_proportion`.
5. Put the shard in the shared datastore and symlink it under `data/`, so sibling
   nodes reuse it instead of re-encoding.
6. Delete the raw footage once latents verify — raw is ~5x the size of latents.

Verify before training: instantiate `SanaWMZipLatentDataset` with your config and pull
one batch. A shard that silently contributes zero samples is the usual failure.
