import xarray as xr

path = r"C:\Users\Prana\Downloads\NASA\Real Data\countersar_outputs.nc"

ds = xr.open_dataset(path)

print("\nDATASET SUMMARY")
print(ds)

print("\nDIMENSIONS")
print(ds.dims)

print("\nVARIABLES")
for name in ds.data_vars:
    print(f"{name}: {ds[name].shape} | {ds[name].dtype}")

print("\nCOORDINATES")
for name in ds.coords:
    print(f"{name}: {ds[name].shape}")

print("\nGLOBAL ATTRIBUTES")
for key, value in ds.attrs.items():
    print(f"{key}: {value}")

print("\nSAMPLE DATA")
for name in ds.data_vars:
    print(f"\n{name}")
    print(ds[name].isel(time=0).values if "time" in ds[name].dims else ds[name].values)

ds.close()