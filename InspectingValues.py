import xarray as xr
import numpy as np

file_path = r"C:\Users\Prana\Downloads\NASA\Evaluation Report Things\dummy_step4_full_outputs.nc"

ds = xr.open_dataset(file_path)

print("Coordinate spacing:")
print("X:", np.diff(ds.x.values[:5]))
print("Y:", np.diff(ds.y.values[:5]))

print("\nVariable statistics:")

for name in ds.data_vars:
    data = ds[name].values

    print("\n", name)
    print("Minimum:", np.nanmin(data))
    print("Maximum:", np.nanmax(data))
    print("Mean:", np.nanmean(data))
    print("NaN count:", np.isnan(data).sum())

print("\nVerdict distribution:")

for date in ds.time.values:
    values = ds["verdicts"].sel(time=date).values
    unique, counts = np.unique(values, return_counts=True)

    print("\nDate:", str(date)[:10])

    for value, count in zip(unique, counts):
        print("Verdict code:", value, "| Pixels:", count)

print("\nSample pixel values:")

for date in ds.time.values:
    print("\nDate:", str(date)[:10])

    for name in ds.data_vars:
        print(name, ":", ds[name].sel(time=date).isel(y=100, x=100).values)