Python version of https://github.com/dronemapper-io/dji-dewarp



``` python dewarp.py JPG input_dir output_dir```


```
Example from Phantom 4 RTK

	Dewarp Data                     : 2018-09-04;3678.870000000000,3671.840000000000,10.100000000000,27.290000000000,-0.268652000000,0.114663000000,0.000015268800,-0.000046070700,-0.035026100000
	Dewarp Flag                     : 0

	Date and Time, fx, fy, cx, cy, k1, k2, p1, p2, k3
```

#### References

https://stackoverflow.com/questions/53511464/undistort-dji-phantom-4-rtk-image-by-exifxmp-data
https://droneshopperth.com.au/product/dji-phantom-4-rtk/

```
Disable the distortion correction would have Phantom 4 RTK’s camera capture the unedited image with the fisheye effect.
The lens distortion parameters were all pre-measured.
These parameters would all be saved in “DewarpData” under the XMP field for every image taken.
You can also input these parameters manually to calibrate the image distortion with the third-party software.
```

