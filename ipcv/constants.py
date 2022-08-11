import numpy

# Remapping interpolation types
INTER_NEAREST = 0   # Nearest-neighbor interpolation
INTER_LINEAR = 1    # Bilinear interpolation
INTER_CUBIC = 2     # Bicubic interpolation over a 4x4 pixel neighborhood

# Border types (image boundaries denoted by '|')
BORDER_CONSTANT = 0      # iiiiii|abcdefgh|iiiiii
BORDER_REPLICATE = 1     # aaaaaa|abcdefgh|hhhhhh
BORDER_REFLECT = 2       # fedcba|abcdefgh|hgfedc
BORDER_REFLECT_101 = 4   # gfedcb|abcdefgh|gfedcb
BORDER_WRAP = 3          # cdefgh|abcdefgh|abcdef

# Destination depths
IPCV_8U = numpy.uint8      # 8-bit unsigned int
IPCV_8S = numpy.int8       # 8-bit signed int
IPCV_16U = numpy.uint16    # 16-bit unsigned int
IPCV_16S = numpy.int16     # 16-bit signed int
IPCV_32S = numpy.int32     # 32-bit signed int
IPCV_32F = numpy.float32   # 32-bit float (single precision)
IPCV_64F = numpy.float64   # 64-bit float (double precision)

# Frequency filter shapes
IPCV_IDEAL = 0         # Ideal
IPCV_BUTTERWORTH = 1   # Butterworth
IPCV_GAUSSIAN = 2      # Gaussian

# libdc1394 types and definitions
DC1394_VIDEO_MODE_160x120_YUV444=64
DC1394_VIDEO_MODE_320x240_YUV422=65
DC1394_VIDEO_MODE_640x480_YUV411=66
DC1394_VIDEO_MODE_640x480_YUV422=67
DC1394_VIDEO_MODE_640x480_RGB8=68
DC1394_VIDEO_MODE_640x480_MONO8=69
DC1394_VIDEO_MODE_640x480_MONO16=70
DC1394_VIDEO_MODE_800x600_YUV422=71
DC1394_VIDEO_MODE_800x600_RGB8=72
DC1394_VIDEO_MODE_800x600_MONO8=73
DC1394_VIDEO_MODE_1024x768_YUV422=74
DC1394_VIDEO_MODE_1024x768_RGB8=75
DC1394_VIDEO_MODE_1024x768_MONO8=76
DC1394_VIDEO_MODE_800x600_MONO16=77
DC1394_VIDEO_MODE_1024x768_MONO16=78
DC1394_VIDEO_MODE_1280x960_YUV422=79
DC1394_VIDEO_MODE_1280x960_RGB8=80
DC1394_VIDEO_MODE_1280x960_MONO8=81
DC1394_VIDEO_MODE_1600x1200_YUV422=82
DC1394_VIDEO_MODE_1600x1200_RGB8=83
DC1394_VIDEO_MODE_1600x1200_MONO8=84
DC1394_VIDEO_MODE_1280x960_MONO16=85
DC1394_VIDEO_MODE_1600x1200_MONO16=86
DC1394_VIDEO_MODE_EXIF=87
DC1394_VIDEO_MODE_FORMAT7_0=88
DC1394_VIDEO_MODE_FORMAT7_1=89
DC1394_VIDEO_MODE_FORMAT7_2=90
DC1394_VIDEO_MODE_FORMAT7_3=91
DC1394_VIDEO_MODE_FORMAT7_4=92
DC1394_VIDEO_MODE_FORMAT7_5=93
DC1394_VIDEO_MODE_FORMAT7_6=94
DC1394_VIDEO_MODE_FORMAT7_7=95
