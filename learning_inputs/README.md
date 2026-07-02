# Example image provenance

The feature-accentuation learning notebook uses local copies of the public
Faccent example photographs so it has no runtime dependency on Faccent code or
files:

- `iguana.jpg`, SHA-256 `976980e1e49b15a1a87f9ff22c40ca66f8166bcd1b09807b5220af6f2df1602a`
- `fox.jpg`, SHA-256 `cc908e850f84f7363e67712d697f8d10a6ea94db6e7b9ff61d1eccf9de92a01e`

Source: <https://github.com/chrishamblin7/faccent/tree/main/test_images>

Only the photographs are retained as example inputs. DreamLens does not import
the Faccent package or its model implementation. The source repository states
an MIT license for its software but does not separately identify a photographer
or license for these two photographs. They are therefore not covered by the
DreamLens Apache-2.0 license and are not included in the PyPI wheel or source
distribution. Downstream redistributors should verify the image rights or
replace them with appropriately licensed examples.
