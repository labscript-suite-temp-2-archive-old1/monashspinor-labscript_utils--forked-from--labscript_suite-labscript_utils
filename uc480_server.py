import uc480
import time
import numpy as np
from camera_server import CameraServer
import h5py
import labscript_utils.properties

# This file implements the protocol for a uc480 camera server, which
# BLACS can interface with to control Thorlabs cameras.
# Requires uc480 module by Daniel R. Dietze:
# https://ddietze.github.io/Py-Hardware-Support/uc480.html
# https://github.com/ddietze/Py-Hardware-Support/tree/master/uc480

class uc480CameraServer(CameraServer):
    """Subclass of CameraServer for Thorlabs uc480 cameras.
    
    transition_to_buffered: Reads /devices/<camera_name>/EXPOSURES, and 
    device_properties of the connection table in the experiment shot file and
    Based on the serial_number provided at instatiation, a uc480 camera instance
    is defined and used to acquire an image after a delay specified in the
    devices added_properties. Prior to acquisition, exposure_time and gain are
    set, and the actual values for these parameters are polled from the camera.

    transition_to_static: Image and attributes are saved to the experiment 
    h5 file.

    abort: Not implemented.
    """

    def __init__(self, camera_name, serial_number, *args, **kwargs):
        super(uc480CameraServer, self).__init__(*args, **kwargs)
        self.camera_name = camera_name
        self.serial_number = serial_number
        self.imageify = True
        # Per shot attributes
        self.camera = None
        self.delay = None
        self.imgs = []


    def transition_to_buffered(self, h5_filepath):
        self.n_images = 0

        # Parse the h5 file for number of exposures and camera properties
        with h5py.File(h5_filepath) as h5_file:
            group = h5_file['devices'][self.camera_name]
            if not 'EXPOSURES' in group:
                print('No camera exposures in this shot.')
                return
            self.exposures = group['EXPOSURES'].value
            self.n_images = len(group['EXPOSURES'])

            # Read camera properties from the h5 file
            self.device_properties = labscript_utils.properties.get(
                h5_file, self.camera_name, 'device_properties')
            self.added_properties = self.device_properties['added_properties']

        # Delay between closing the h5 file and opening camera
        if 'delay' in self.added_properties:
            self.delay = self.added_properties['delay']
            time.sleep(self.delay)

        # Check if the camera is connected
        self.camera = uc480.uc480()
        serial_numbers = [np.uint64(cam_info.SerNo)
                          for cam_info in self.camera._cam_list.uci]
        try:
            self.camera_id = serial_numbers.index(self.serial_number)
        except ValueError:
            print('Camera not found (SN: {:}). '
                  'Setting exposure number to zero.'.format(self.serial_number))
            self.n_images = 0
            return

        # Connect to the camera based on CameraID determined above
        self.camera.connect(self.camera_id)
        self.bitdepth = self.camera._bitsperpixel

        # Configure camera based on device_properties and added_properties
        # dictionaries
        exposure_time_requested = self.device_properties['exposure_time'] * 1e3
        if exposure_time_requested < self.camera.expmin or exposure_time_requested > self.camera.expmax:
            exposure_time_requested = np.clip(
                exposure_time_requested, self.camera.expmin, self.camera.expmax)
            print('Requested exposure time outside those allowed.'
                  'Coercing to {:.3f}ms'.format(exposure_time_requested))
        self.camera.set_exposure(exposure_time_requested)
        if 'gain' in self.added_properties:
            self.camera.set_gain(self.added_properties['gain'])

        # Capture the images
        print('Capturing image.')
        self.imgs = [self.camera.acquire()]
        print('Mean value: {:}'.format(int(self.imgs[0].mean())))

        # Get camera properties by polling camera
        print('Getting camera properties.')
        self.exposure_time = self.camera.get_exposure()
        self.gain = self.camera.get_gain()
        print('Exposure time (actual): {:.3f}ms'.format(self.exposure_time))
        print('Gain (actual): {:}'.format(self.gain))
        # TODO: Save these as attributes to the data set (below)

        # Disconnect from the camera and close it
        print('Disconnecting from the camera.')
        self.camera.disconnect()
        del self.camera


    def transition_to_static(self, h5_filepath):
        start_time = time.time()
        if self.n_images:
            print('Saving {} images.'.format(self.n_images))
            with h5py.File(h5_filepath) as h5_file:
                image_group = h5_file.require_group(
                    '/images/' + self.device_properties['orientation'])
                image_group.attrs['camera'] = str(self.camera_name)
                image_group.attrs.create(
                    'exposure_time', 1e-3*self.exposure_time, dtype='float32')
                image_group.attrs.create('gain', self.gain, dtype='uint8')
                for i, exposure in enumerate(self.exposures):
                    group = h5_file.require_group(
                        '{:}/{:}'.format(image_group.name, exposure['name']))
                    dset = group.create_dataset(exposure['frametype'], data=self.imgs[i],
                                                dtype='uint{:d}'.format(self.bitdepth),
                                                compression='gzip')
                    if self.imageify:
                        # Specify this dataset should be viewed as an image
                        dset.attrs['CLASS'] = np.string_('IMAGE')
                        dset.attrs['IMAGE_VERSION'] = np.string_('1.2')
                        dset.attrs['IMAGE_SUBCLASS'] = np.string_(
                            'IMAGE_GRAYSCALE')
                        dset.attrs['IMAGE_WHITE_IS_ZERO'] = np.uint8(0)
        else:
            print('transition_to_static: No camera exposures in this shot.\n\n')
            return
        print(self.camera_name + ' saving time: {:.3f}ms\n\n'.format(1e3*(time.time()-start_time)))


    def abort(self):
        pass


if __name__ == '__main__':
    import sys
    try:
        camera_name = sys.argv[1]
    except IndexError:
        print('Call me with the name of a camera as defined in BLACS.')
        # sys.exit(0)
        camera_name = 'thor_dcc1545m_0'
        print('Using {:} by default.'.format(camera_name))

    from labscript_utils.labconfig import LabConfig
    print('Reading labconfig')
    lc = LabConfig()

    connection_table_path = lc.get('paths', 'connection_table_h5')
    print('Getting properties of {:} from connection table: {:}'.format(
        camera_name, connection_table_path))
    with h5py.File(connection_table_path) as h5_path:
        device_properties = labscript_utils.properties.get(
            h5_path, camera_name, 'device_properties')
        port = labscript_utils.properties.get(
            h5_path, camera_name, 'connection_table_properties')['BIAS_port']
    serial_number = device_properties['serial_number']

    print('Starting camera server on port %d...' % port)
    server = uc480CameraServer(camera_name, serial_number, port)
    server.shutdown_on_interrupt()
