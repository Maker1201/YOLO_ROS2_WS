import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # --- 参数配置 ---
        # 兼容 .engine (推荐) 或 .onnx
        self.model = YOLO('/home/nvidia/yolo_ros2_ws/models/best.onnx', task='pose') 
        self.bridge = CvBridge()
        
        # --- 订阅与发布 ---
        # 1. 订阅通用图像话题 (请根据实际摄像头驱动修改这个名字)
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        # 2. 发布给 RL/机械臂 的坐标点
        self.pose_pub = self.create_publisher(Point, '/button_pose', 10)
        
        # 3. 发布给 RViz2 观看的图像 (带检测框)
        self.image_pub = self.create_publisher(Image, '/yolo_result_image', 10)
        
        # 4. 发布给 RViz2 的 3D 标记 (Marker)
        self.marker_pub = self.create_publisher(Marker, '/button_marker', 10)
        
        self.get_logger().info('通用 YOLO+RViz 节点已启动，等待图像输入...')

    def image_callback(self, msg):
        # 1. ROS 图像转 OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. 推理
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        # 获取自带画好框的图像
        annotated_frame = results[0].plot() 
        
        for r in results:
            if r.keypoints is not None:
                kpts = r.keypoints.xy.cpu().numpy()
                if len(kpts) > 0 and len(kpts[0]) > 0:
                    x, y = kpts[0][0]  # 假设取第一个目标的第一个关键点
                    
                    if x > 0 and y > 0:
                        # --- 任务 A: 发送坐标给 RL/底盘 ---
                        point_msg = Point(x=float(x), y=float(y), z=0.0)
                        self.pose_pub.publish(point_msg)
                        
                        # --- 任务 B: 发送 3D Marker 给 RViz2 ---
                        self.publish_marker(x, y, msg.header)

        # --- 任务 C: 发送渲染好的画面给 RViz2 ---
        result_img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
        result_img_msg.header = msg.header # 保持时间戳和坐标系一致
        self.image_pub.publish(result_img_msg)

    def publish_marker(self, x, y, header):
        """生成并在 RViz 中显示一个球体代表目标点"""
        marker = Marker()
        # 必须设置 frame_id，通常与摄像头的 frame_id 保持一致
        marker.header.frame_id = header.frame_id if header.frame_id else "camera_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fire_button"
        marker.id = 0
        marker.type = Marker.SPHERE # 形状：球体
        marker.action = Marker.ADD
        
        # 注意：这里是将 2D 像素坐标强行映射到 3D 空间用于演示。
        # 实际抓取时，需要根据深度相机或手眼标定矩阵转换为真实 3D 坐标。
        marker.pose.position.x = float(x) / 100.0  # 缩小比例以适应 RViz 视野
        marker.pose.position.y = float(y) / 100.0
        marker.pose.position.z = 1.0 # 固定深度
        
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.a = 1.0 # 透明度
        marker.color.r = 0.0
        marker.color.g = 1.0 # 绿色
        marker.color.b = 0.0
        
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()