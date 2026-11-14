import xarray as xr

file_path = r"C:\Users\Prana\Downloads\NASA\Evaluation Report Things\dummy_step4_full_outputs.nc"

ds = xr.open_dataset(file_path)

print(ds)
print("\nVariables:")
print(list(ds.data_vars))

print("\nCoordinates:")
print(list(ds.coords))

for name in ds.data_vars:
    print("\n", name)
    print(ds[name].dims)
    print(ds[name].shape)