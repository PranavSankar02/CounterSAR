import xarray as xr
import numpy as np

ds = xr.open_dataset(r"C:\Users\Prana\Downloads\NASA\Evaluation Report Things\dummy_step4_full_outputs.nc")

for var in ds.data_vars:
    data = ds[var].values

    print(f"\n{var}")
    print("Min:", np.nanmin(data))
    print("Max:", np.nanmax(data))
    print("Mean:", np.nanmean(data))
    print("Unique values:", len(np.unique(data)))

print("\nVerdict distribution:")

for i, date in enumerate(ds.time.values):
    values, counts = np.unique(
        ds.verdicts.isel(time=i).values,
        return_counts=True
    )

    total_pixels = ds.verdicts.isel(time=i).size

    print("\nDate:", str(date)[:10])

    for value, count in zip(values, counts):
        print(f"Verdict {value}: {count} pixels ({count / total_pixels * 100:.2f}%)")

print("\nSpatial variation:")

for var in ["observed", "predicted_clim", "predicted_weather"]:
    data = ds[var].values

    print(var)
    print("Date 1 standard deviation:", np.nanstd(data[0]))
    print("Date 2 standard deviation:", np.nanstd(data[1]))

print("\nCoordinate spacing:")
print("X:", np.unique(np.round(np.diff(ds.x.values), 3)))
print("Y:", np.unique(np.round(np.diff(ds.y.values), 3)))
