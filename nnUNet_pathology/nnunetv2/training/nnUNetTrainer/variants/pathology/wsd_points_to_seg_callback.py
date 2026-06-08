import numpy as np
import cv2
from shapely.geometry import Point, MultiPoint, MultiPolygon
import random

class PointsToSegBatchCallback:
    def __init__(self, point_sizes_dict, spacing):
        self.point_sizes_dict = {int(k): int(v) for k, v in point_sizes_dict.items()}
        self.spacing = spacing
        # Precompute distances in pixels for the buffer operations to avoid repeated calculations
        self.buffer_distances_px = {value: int(round(size / self.spacing)) for value, size in self.point_sizes_dict.items()}
    
    def dist_to_px(self, dist):
        """ distance in um (or rather same unit as the spacing) """
        dist_px = int(round(dist / self.spacing))
        return dist_px

    def find_coords_for_values(self, arr, values):
        # Using numpy to find coordinates for multiple values at once, reducing Python loop overhead
        unique_values = np.unique(arr)
        values_to_find = [value for value in values if value in unique_values]
        coords_dict = {value: np.argwhere(arr == value) for value in values_to_find}
        return coords_dict

    def array_to_points_with_buffer(self, array):
        result_dict = {}
        coords_dict = self.find_coords_for_values(array, list(self.point_sizes_dict.keys()))
        for value, coords in coords_dict.items():
            buffer_distance_px = self.buffer_distances_px[value]
            for coord in coords:
                point = Point(coord[1], coord[0])
                buffered_point = point.buffer(buffer_distance_px).simplify(1)
                result_dict.setdefault(value, []).append(buffered_point)

        return result_dict

    def seg_mask_from_points_dict(self, points_dict, input_array):
        result_array = np.copy(input_array)
        for class_value in points_dict:
            for point in points_dict[class_value]:
                coords = np.array(point.exterior.coords)
                coords = np.round(coords).astype(np.int32)
                coords = coords.reshape((-1, 1, 2))
                cv2.fillPoly(result_array, [coords], color=int(class_value))
        return result_array

    def point_mask_to_seg_mask(self, mask_patch):
        points_dict = self.array_to_points_with_buffer(mask_patch)
        seg_mask = self.seg_mask_from_points_dict(points_dict, mask_patch)
        return seg_mask

    def __call__(self, x_batch, y_batch):
        transformed_y_batch = []
        for mask_patch in y_batch:
            seg_mask = self.point_mask_to_seg_mask(mask_patch)
            transformed_y_batch.append(seg_mask)
        return x_batch, np.array(transformed_y_batch)

### NEED TO DO LOOP OVER POINTS INSTEAD OF MULTIPOINT
# class PointsToSegBatchCallbackCentroid(PointsToSegBatchCallback):
#     def __init__(self, point_sizes_dict, spacing, centroid_size=1.5, centroid_value=None):
#         super().__init__(point_sizes_dict, spacing)
#         self.centroid_size = centroid_size
#         self.centroid_value = centroid_value if centroid_value is not None else max(int(k) for k in self.point_sizes_dict.keys()) + 1

#     def array_to_points_with_buffer(self, array):
#         # Same as super
#         result_dict = {}
#         coords_dict = self.find_coords_for_values(array, list(self.point_sizes_dict.keys()))
        
#         all_points = []
#         for value, coords in coords_dict.items():
#             buffer_distance = self.point_sizes_dict[value]
#             buffer_distance_px = self.dist_to_px(buffer_distance)
            
#             points = [Point(coord[1], coord[0]) for coord in coords]
#             multi_point = MultiPoint(points)
#             buffered_multi_point = multi_point.buffer(buffer_distance_px).simplify(1)
            
#             all_points.extend(points)
#             result_dict.setdefault(value, []).extend(self.geom_check(buffered_multi_point))
        
#         # Add centroids
#         buffer_distance_px = self.dist_to_px(self.centroid_size)
#         multi_point_all = MultiPoint(all_points)
#         buffered_multi_point_all = multi_point_all.buffer(buffer_distance_px).simplify(1)
#         result_dict[self.centroid_value] = self.geom_check(buffered_multi_point_all)

#         return result_dict

### NEED TO DO LOOP OVER POINTS INSTEAD OF MULTIPOINT
# class PointsToSegBatchCallbackOuterBorder(PointsToSegBatchCallback):
#     """
#     border gets new value and a single size that is added to the point's size
#     """
#     # TODO: check where to remove multipoint and use for loop to buffer to prevent filling holes in the buffered polygons
#     def __init__(self, point_sizes_dict, spacing, border_size, border_value=None):
#         super().__init__(point_sizes_dict, spacing)
#         self.border_size = border_size
#         self.border_value = border_value if border_value is not None else max(int(k) for k in self.point_sizes_dict.keys()) + 1

#     def array_to_points_with_buffer(self, array):
#         result_dict = {}
#         result_dict.setdefault(self.border_value, [])

#         coords_dict = self.find_coords_for_values(array, list(self.point_sizes_dict.keys()))

#         all_points = []
#         for value, coords in coords_dict.items():
#             buffer_distance = self.point_sizes_dict[value]
#             buffer_distance_px = self.dist_to_px(buffer_distance)
            
#             buffer_distance_border = self.border_size
#             buffer_distance_px_border = self.dist_to_px(buffer_distance + buffer_distance_border)
            
#             points = [Point(coord[1], coord[0]) for coord in coords]
#             multi_point = MultiPoint(points)
#             buffered_multi_point = multi_point.buffer(buffer_distance_px).simplify(1)

#             all_points.extend(points)
#             result_dict.setdefault(value, []).extend(self.geom_check(buffered_multi_point))

#         multi_point_all = MultiPoint(all_points)
#         buffered_multi_point_border = multi_point_all.buffer(buffer_distance_px_border).simplify(1)
#         result_dict[self.border_value].extend(self.geom_check(buffered_multi_point_border))

#         return result_dict

class PointsToSegBatchCallbackInnerBorderTillCentroid(PointsToSegBatchCallback):
    """
    border gets new value and ranges from the outer border to the centroid size, centroid gets class value
    """
    def __init__(self, point_sizes_dict, spacing, centroid_size=1.5, border_value=None):
        super().__init__(point_sizes_dict, spacing)
        self.centroid_size = centroid_size
        self.border_value = border_value if border_value is not None else max(int(k) for k in self.point_sizes_dict.keys()) + 1

    def array_to_points_with_buffer(self, array):
        result_dict = {}
        coords_dict = self.find_coords_for_values(array, list(self.point_sizes_dict.keys()))

        for value, coords in coords_dict.items():
            # Precompute the buffer distances for both border and centroids
            buffer_distance_px = self.buffer_distances_px[value]  # Border size
            centroid_buffer_distance_px = self.dist_to_px(self.centroid_size)  # Centroid size

            # Create buffered points for border using list comprehension
            border_buffered_points = [
                Point(coord[1], coord[0]).buffer(buffer_distance_px).simplify(1)
                for coord in coords
            ]
            result_dict.setdefault(self.border_value, []).extend(border_buffered_points)

            # Create buffered points for centroids using list comprehension
            centroid_buffered_points = [
                Point(coord[1], coord[0]).buffer(centroid_buffer_distance_px).simplify(1)
                for coord in coords
            ]
            result_dict.setdefault(value, []).extend(centroid_buffered_points)

        return result_dict
    
### ADVANCED CALLBACK, DEALS WITH OVERLAPPING CENTROIDS, BUT SLOWER
class PointsToSegBatchCallbackDoubleBorderRandomOrder(PointsToSegBatchCallback):
    def __init__(self, point_sizes_dict, spacing, centroid_size=1.5, border_value=None):
        super().__init__(point_sizes_dict, spacing)
        self.centroid_size = centroid_size
        self.small_border_size = 1
        self.border_value = border_value if border_value is not None else max(int(k) for k in self.point_sizes_dict.keys()) + 1

    def flatten_one_deep(self, nested_list):
        return [item for sublist in nested_list for item in sublist]

    def array_to_points_with_buffer(self, array):
        point_with_small_border_list = []
        big_border_list = []
        coords_dict = self.find_coords_for_values(array, list(self.point_sizes_dict.keys()))
        
        for value, coords in coords_dict.items():
            buffer_distance = self.point_sizes_dict[value]
            buffer_distance_px = self.dist_to_px(buffer_distance)

            points = [Point(coord[1], coord[0]) for coord in coords]
            # First big background borders
            for point in points:
                buffered_point = point.buffer(buffer_distance_px).simplify(1)
                big_border_list.append([buffered_point, self.border_value])

            # Then the small border and the point
            for point in points:
                buffered_small_border = point.buffer(self.centroid_size + self.small_border_size).simplify(1)
                buffered_point = point.buffer(self.centroid_size).simplify(1)
                point_with_small_border_list.extend([[[buffered_small_border, self.border_value], [buffered_point, value]]]) # we put the small border and the point together in a list so they will be shuffeled together

        random.shuffle(point_with_small_border_list)
        point_with_value_list = big_border_list + self.flatten_one_deep(point_with_small_border_list)

        return point_with_value_list

    def seg_mask_from_points_list(self, points_list, input_array):
        result_array = np.copy(input_array)
        for point, class_value in points_list: 
            coords = np.array(point.exterior.coords)
            coords = np.round(coords).astype(np.int32)
            coords = coords.reshape((-1, 1, 2))
            cv2.fillPoly(result_array, [coords], color=int(class_value))
        return result_array
    
    def point_mask_to_seg_mask(self, mask_patch):
        points_dict = self.array_to_points_with_buffer(mask_patch)
        seg_mask = self.seg_mask_from_points_list(points_dict, mask_patch)
        return seg_mask
