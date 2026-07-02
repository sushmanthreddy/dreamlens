# Natural-image Fourier magnitude

`clean_decorrelated.npy` is DreamLens's packaged RGB natural-image magnitude
spectrum. It is sourced from the same upstream spectrum URL used by the MaCo
path:

<https://storage.googleapis.com/serrelab/loupe/spectrums/imagenet_decorrelated.npy>

DreamLens uses this one packaged asset for both default MaCo initialization and the optional
`magnitude_source="imagenet"` feature-accentuation mode.

- Shape: `(3, 512, 257)` (`rfft2` layout for a 512 × 512 RGB image)
- Data type: `float32`
- SHA-256: `a4810ea049ef9a0fe4e3f26660188e53222281879b333e8fd61377f7491aafc8`

The seeded Faccent `fourier_phase` path remains
`magnitude_source="image"` by default, because the reference implementation
derives magnitude and phase from the input image when an initial image is
supplied. Faccent's main/default accentuation path instead uses the full
seed-initialized complex Fourier parameterization.
