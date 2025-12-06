# src/utils/camera_model.py
import math

def fx_from_fov(width_px, fov_x_deg):
    return (width_px / 2.0) / math.tan(math.radians(fov_x_deg) / 2.0)

def fy_from_fov(height_px, fov_y_deg):
    return (height_px / 2.0) / math.tan(math.radians(fov_y_deg) / 2.0)

def make_cam_params(width=640, height=480, fov_x=30.0, fov_y=None, cx=None, cy=None):
    if fov_y is None:
        fov_y = 2.0 * math.degrees(math.atan((height/width) * math.tan(math.radians(fov_x)/2.0)))
    fx = fx_from_fov(width, fov_x)
    fy = fy_from_fov(height, fov_y)
    if cx is None:
        cx = width / 2.0
    if cy is None:
        cy = height / 2.0
    return {
        'width': int(width),
        'height': int(height),
        'fov_x': float(fov_x),
        'fov_y': float(fov_y),
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy)
    }

def direction_to_pixel(vec_cam, cam_params):
    X, Y, Z = float(vec_cam[0]), float(vec_cam[1]), float(vec_cam[2])
    if Z <= 0:
        return None
    x_n = X / Z
    y_n = Y / Z
    u = cam_params['fx'] * x_n + cam_params['cx']
    v = cam_params['fy'] * y_n + cam_params['cy']
    if not (0.0 <= u < cam_params['width'] and 0.0 <= v < cam_params['height']):
        return None
    return (u, v)
