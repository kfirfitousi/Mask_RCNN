import os.path
from matplotlib import pyplot as plt
from vidgear.gears import CamGear, WriteGear
import mrcnn.model as modellib
from mrcnn import visualize
from f1 import F1Config
import cv2

class InferenceConfig(F1Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8


class VideoInference:
    def __init__(self, model, video, logs="logs"):
        self.model = model
        self.video = video
        self.logs = logs
        self.class_names = [
            "BG",
            "Alfa Romeo",
            "Alpha Tauri",
            "Alpine",
            "Aston Martin",
            "Ferarri",
            "Haas",
            "McLaren",
            "Mercedes",
            "Red Bull",
            "Williams"
        ]

    def infer(self):
        if self.model is None:
            print("must provide weights")
            return

        if self.video is None:
            print("must provide input video path")
            return

        # Configurations
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=self.logs, config=config)

        # Load weights trained on the F1 dataset
        if self.model.lower() == "last":
            self.model = model.find_last()

        model.load_weights(self.model, by_name=True)

        # create video reader and writer
        stream = CamGear(source=self.video).start()
        writer = WriteGear(output=os.path.dirname(os.path.abspath(self.video)) + 'output.mp4')

        # frame dimensions
        width, height = None, None

        while True:
            # read frame
            frame = stream.read()

            if frame is None:
                break

            # find video dimensions
            width = frame.shape[1] if width is None else width
            height = frame.shape[0] if height is None else height

            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run detection
            results = model.detect([frame], verbose=1)

            # Visualize results
            r = results[0]
            fig, ax = plt.subplots(1, figsize=(16, 16))
            visualize.display_instances(frame, r['rois'], r['masks'],
                                        r['class_ids'], self.class_names,
                                        r['scores'], show_mask=False, ax=ax)
            plt.savefig('frame.png', bbox_inches='tight',
                        pad_inches=-0.5, orientation='landscape')
            plt.close(fig)
            frame = cv2.imread('frame.png')

            # resize frame to original video dimensions
            frame = cv2.resize(frame, (width, height), cv2.INTER_AREA)

            # write frame to output
            writer.write(frame)

            # check for 'q' key if pressed
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # close output window
        cv2.destroyAllWindows()

        # safely close video stream
        stream.stop()

        # safely close writer
        writer.close()

