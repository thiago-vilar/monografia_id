import cv2
import numpy as np
from matplotlib import pyplot as plt
import stag
from rembg import remove

class DetectMedicine:
    def __init__(self, image_path, neg_image_path, stag_id):
        self.image_path = image_path
        self.neg_image_path = neg_image_path
        self.stag_id = stag_id
        self.pixel_size_mm = None  
        self.load_images()
        self.corners = None
        self.scan_areas = {}
        self.preload_rembg()

    def preload_rembg(self):
        dummy_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        try:
            is_success, buffer = cv2.imencode(".png", dummy_image)
            if is_success:
                remove(buffer.tobytes())
            print("rembg model preloaded successfully.")
        except Exception as e:
            print(f"Failed to preload rembg model: {e}")

    def load_images(self):
        self.image = cv2.imread(self.image_path)
        self.neg_image = cv2.imread(self.neg_image_path)
        if self.image is None or self.neg_image is None:
            raise ValueError("One or both images could not be loaded. Please check the paths.")

    def detect_stag(self, image):
        config = {'libraryHD': 17, 'errorCorrection': 0}
        corners, ids, _ = stag.detectMarkers(image, **config)
        if ids is not None and self.stag_id in ids:
            index = np.where(ids == self.stag_id)[0][0]
            self.corners = corners[index].reshape(-1, 2)
            self.calculate_pixel_size_mm()
            return self.corners
        print(f"Marker with ID {self.stag_id} not found in one of the images.")
        return None

    def calculate_pixel_size_mm(self):
        if self.corners is not None:
            width_px = np.max(self.corners[:, 0]) - np.min(self.corners[:, 0])
            self.pixel_size_mm = 20.0 / width_px
        else:
            self.pixel_size_mm = None
            print("Failed to detect corners to calculate pixel size.")

    def homogenize_image_based_on_corners(self, image, corners):
        if corners is None:
            print("Corners not detected.")
            return None
        x, y, w, h = cv2.boundingRect(corners.astype(np.float32))
        aligned_corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype='float32')
        transform_matrix = cv2.getPerspectiveTransform(corners, aligned_corners)
        return cv2.warpPerspective(image, transform_matrix, (image.shape[1], image.shape[0]))

    def display_scan_area_by_markers(self, image):
        if self.pixel_size_mm is None:
            print("Pixel size not set, cannot calculate scan area.")
            return None
        if image is None:
            print("Homogenized image is not available.")
            return None
        if self.corners is None:
            print("Corners are not detected.")
            return None
        corners_int = self.corners.astype(int)
        centroid_x = int(np.mean(corners_int[:, 0]))
        centroid_y = int(np.mean(corners_int[:, 1]))
        cv2.putText(image, f'ID:{self.stag_id}', (centroid_x + 45, centroid_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        marker_width_px = np.max(corners_int[:, 0]) - np.min(corners_int[:, 0])
        pixel_size_mm = marker_width_px / 20.0  
        crop_width = int(35 * pixel_size_mm)
        crop_height = int(70 * pixel_size_mm)
        crop_y_adjustment = int(15 * pixel_size_mm)

        x_min = max(centroid_x - crop_width, 0)
        x_max = min(centroid_x + crop_width, image.shape[1])
        y_min = max(centroid_y - crop_height - crop_y_adjustment, 0)
        y_max = min(centroid_y - crop_y_adjustment, image.shape[0])

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        self.scan_areas[self.stag_id] = (x_min, x_max, y_min, y_max)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image with Scan Area')
        plt.show()

        return (x_min, x_max, y_min, y_max)

    def crop_scan_area(self, image, crop_coords):
        x_min, x_max, y_min, y_max = crop_coords
        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    def remove_background(self, image_np_array):
        is_success, buffer = cv2.imencode(".png", image_np_array)
        if not is_success:
            raise ValueError("Failed to encode image for background removal.")
        output_image = remove(buffer.tobytes())
        img_med = cv2.imdecode(np.frombuffer(output_image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img_med is None:
            raise ValueError("Failed to decode processed image.")
        return img_med

    def create_mask(self, img):
        if img.shape[2] == 4:
            img = img[:, :, :3]  # Remove alpha channel
        lower_bound = np.array([0, 0, 10])
        upper_bound = np.array([255, 255, 255])
        mask = cv2.inRange(img, lower_bound, upper_bound)
        mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)
        mask_rgba[:, :, 3] = mask 
        return mask

    def find_and_draw_contours(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
            mask_with_contours[:, :, 3] = mask
            cv2.drawContours(mask_with_contours, [largest_contour], -1, (0, 255, 0, 255), 1)
            return mask_with_contours, largest_contour
        else:
            return None

    def medicine_measures(self, img_med, largest_contour):
        if largest_contour is None or self.pixel_size_mm is None:
            print("No valid contour to measure or pixel size not set.")
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        width_mm = w * self.pixel_size_mm 
        height_mm = h * self.pixel_size_mm 
        area_mm2 = width_mm * height_mm  
        mask_area_mm2 = cv2.contourArea(largest_contour) * (self.pixel_size_mm ** 2)
        background_area_mm2 = area_mm2 - mask_area_mm2  

        print(f"Bounding box at ({x}, {y}), width: {width_mm:.1f}mm, height: {height_mm:.1f}mm, Bounding Box Area: {area_mm2:.2f} mm^2")
        print(f"Largest Contour Area: {mask_area_mm2:.2f} mm^2")
        print(f"Background Area (Bounding Box - Largest Contour Area): {background_area_mm2:.2f} mm^2")

        measured_img = img_med.copy()
        cv2.rectangle(measured_img, (x, y), (x + w, y + h), (0, 255, 0), 1)  
        # Adiciona texto com a largura e a altura
        cv2.putText(measured_img, f"Width: {width_mm:.1f}mm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(measured_img, f"Height: {height_mm:.1f}mm", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return (width_mm, height_mm, area_mm2, mask_area_mm2, background_area_mm2), measured_img

    def display_image(self, image, title):
        if image.ndim == 2 or image.shape[2] == 1:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.show()

    def classify_medicine_type(self, width_mm, area_mm2):
        # Classify based on provided thresholds
        if 10.99 <= width_mm <= 17.9 and 155.22 <= area_mm2 <= 1000: # extent 67 62
            return "Ampoule"
        elif 18.86 <= width_mm <= 25.67 and 276.69 <= area_mm2 <= 1000: # extent 78
            return "Flask"
        elif 48.61 <= width_mm <= 53.91 and 1964.49 <= area_mm2 <= 4000: # extent 88
            return "Pill"
        else:
            return "Unknown"

    def process_image(self):
        corners_image = self.detect_stag(self.image)
        if corners_image is None:
            return

        homogenized_image = self.homogenize_image_based_on_corners(self.image, corners_image)
        crop_coords_image = self.display_scan_area_by_markers(homogenized_image)
        cropped_image = self.crop_scan_area(homogenized_image, crop_coords_image)
        self.display_image(cropped_image, 'Medicine Image')

        # Resize the negative image to match the size of the cropped image
        cropped_neg_image = cv2.resize(self.neg_image, (cropped_image.shape[1], cropped_image.shape[0]))

        # Subtract the resized negative image from the cropped image
        subtracted_image = cv2.subtract(cropped_image, cropped_neg_image)
        self.display_image(subtracted_image, 'Subtracted Image')

        remove_background = self.remove_background(subtracted_image)
        self.display_image(remove_background, 'Removed Background')

        mask = self.create_mask(remove_background)
        self.display_image(mask, 'Mask')

        area_mm2 = np.count_nonzero(mask) * (self.pixel_size_mm ** 2)
        print(f"Area of the mask in mm^2: {area_mm2:.2f}")

        mask_with_contours, largest_contour = self.find_and_draw_contours(mask)
        if largest_contour is not None:
            contour_img = np.zeros_like(remove_background)
            cv2.drawContours(contour_img, [largest_contour], -1, (0, 255, 0), 1)
            self.display_image(contour_img, 'Largest Contour by Mask')

            measures, measured_medicine = self.medicine_measures(remove_background, largest_contour)
            if measured_medicine is not None:
                self.display_image(measured_medicine, 'Measured Medicine')
                medicine_type = self.classify_medicine_type(measures[0], measures[2])
                print(f"Medicine Type: {medicine_type}")

def main():
    stag_id=0
    image_path = "./utils/thiago_fotos_10_up_lighton/img_0_008.jpg"
    neg_image_path = "./neg_image_70x70.jpg"
    detector = DetectMedicine(image_path, neg_image_path, stag_id)
    detector.process_image()

if __name__ == "__main__":
    main()
