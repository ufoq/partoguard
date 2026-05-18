import os
import random
import cv2

def generate_demos() -> None:
    who_chart_path = 'data/harvested/open_access_papers/extracted_figures/bmc2013_p3_fig1.jpeg'
    output_dir = 'android/app/src/main/assets/demo_partographs'
    
    base_img = cv2.imread(who_chart_path)
    if base_img is None:
        raise ValueError("Could not read image")
    
    crop_y1 = 320  
    crop_y2 = 680  
    crop_x1 = 60   
    crop_x2 = 850  
    
    cropped_base = base_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    x_start = 130
    y_start = 328
    px_per_cm = 31.3
    px_per_hour = 49.67

    def px(h: float, cm: float) -> tuple[int, int]:
        return (int(x_start + h * px_per_hour), int(y_start - cm * px_per_cm))

    def draw_x(img: cv2.typing.MatLike, pt: tuple[int, int], rng: random.Random) -> None:
        cx, cy = pt
        sz = rng.randint(7, 10)
        thickness = rng.randint(2, 3)
        color = (120, 20, 20)
        cv2.line(img, (cx - sz, cy - sz), (cx + sz, cy + sz), color, thickness, cv2.LINE_AA)
        cv2.line(img, (cx + sz, cy - sz), (cx - sz, cy + sz), color, thickness, cv2.LINE_AA)
        
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(42)

    cv2.imwrite(os.path.join(output_dir, 'demo_blank.png'), cropped_base)

    img_normal = cropped_base.copy()
    points_normal = [(0.0, 4.5), (1.0, 6.5), (2.0, 8.5), (3.0, 10.0)]
    for h, cm in points_normal:
        draw_x(img_normal, px(h, cm), rng)
    cv2.imwrite(os.path.join(output_dir, 'demo_normal.png'), img_normal)

    img_alert = cropped_base.copy()
    points_alert = [(0.0, 4.0), (1.0, 4.5), (2.0, 5.2), (3.0, 6.0), (4.0, 6.5), (5.0, 7.5), (6.0, 8.5)]
    for h, cm in points_alert:
        draw_x(img_alert, px(h, cm), rng)
    cv2.imwrite(os.path.join(output_dir, 'demo_alert.png'), img_alert)

    img_action = cropped_base.copy()
    points_action = [(0.0, 4.0), (1.0, 4.3), (2.0, 4.7), (3.0, 5.0), (4.0, 5.2), (5.0, 5.3), (6.0, 5.4), (7.0, 5.5), (8.0, 5.5), (9.0, 5.6)]
    for h, cm in points_action:
        draw_x(img_action, px(h, cm), rng)
    cv2.imwrite(os.path.join(output_dir, 'demo_action.png'), img_action)

if __name__ == '__main__':
    generate_demos()
