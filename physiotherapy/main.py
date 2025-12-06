import cv2
import math
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import mediapipe as mp

# 🟩 اضافه برای فارسی
import arabic_reshaper
from bidi.algorithm import get_display

def fix_farsi(text):
    """درست کردن چسبندگی و جهت متن فارسی"""
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
BODY_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# Load KV file
Builder.load_file("myapp.kv")

# Video capture
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    ba = [a[0]-b[0], a[1]-b[1]]
    bc = [c[0]-b[0], c[1]-b[1]]
    cos_angle = (ba[0]*bc[0] + ba[1]*bc[1]) / (math.hypot(*ba)*math.hypot(*bc)+1e-6)
    angle = math.degrees(math.acos(cos_angle))
    return int(angle)

def angle_color(angle, min_val, max_val):
    # تغییر رنگ زاویه به سفید ثابت
    return 1, 1, 1, 1

# --- Screens ---
class MainScreen(Screen):
    def on_kv_post(self, base_widget):
        self.ids.btn_elbow.text = fix_farsi("آرنج")
        self.ids.btn_knee.text = fix_farsi("زانو")
        self.ids.btn_shoulder.text = fix_farsi("شانه")

class ElbowScreen(Screen):
    def on_kv_post(self, base_widget):
        self.ids.btn_back_elbow.text = fix_farsi("بازگشت")

    def on_enter(self):
        Clock.schedule_interval(self.update, 1/30)
    def on_leave(self):
        Clock.unschedule(self.update)
    def update(self, dt):
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        left_angle = right_angle = 0

        if results.pose_landmarks:
            left_angle = calculate_angle(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            )
            right_angle = calculate_angle(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            )
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, BODY_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2)
            )

        # Label ها و ProgressBar
        self.ids.left_elbow_label.text = fix_farsi(f"زاویه آرنج چپ: {left_angle}")
        self.ids.left_elbow_label.color = angle_color(left_angle,30,150)
        self.ids.right_elbow_label.text = fix_farsi(f"زاویه آرنج راست: {right_angle}")
        self.ids.right_elbow_label.color = angle_color(right_angle,30,150)
        self.ids.left_elbow_bar.value = left_angle
        self.ids.right_elbow_bar.value = right_angle

        buf = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

class KneeScreen(Screen):
    def on_kv_post(self, base_widget):
        self.ids.btn_back_knee.text = fix_farsi("بازگشت")

    def on_enter(self):
        Clock.schedule_interval(self.update, 1/30)
    def on_leave(self):
        Clock.unschedule(self.update)
    def update(self, dt):
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        left_angle = right_angle = 0

        if results.pose_landmarks:
            left_angle = calculate_angle(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
            )
            right_angle = calculate_angle(
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE],
                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
            )
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, BODY_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2)
            )

        self.ids.left_knee_label.text = fix_farsi(f"زاویه زانوی چپ: {left_angle}")
        self.ids.left_knee_label.color = angle_color(left_angle,70,170)
        self.ids.right_knee_label.text = fix_farsi(f"زاویه زانوی راست: {right_angle}")
        self.ids.right_knee_label.color = angle_color(right_angle,70,170)
        self.ids.left_knee_bar.value = left_angle
        self.ids.right_knee_bar.value = right_angle

        buf = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

class ShoulderScreen(Screen):
    def on_kv_post(self, base_widget):
        self.ids.btn_back_shoulder.text = fix_farsi("بازگشت")

    def on_enter(self):
        Clock.schedule_interval(self.update,1/30)
    def on_leave(self):
        Clock.unschedule(self.update)
    def update(self,dt):
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, BODY_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(166,77,255), thickness=2)
            )
        buf = cv2.flip(frame,0).tobytes()
        texture = Texture.create(size=(frame.shape[1],frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.cam_feed.texture = texture

# Screen Manager
class MyApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(MainScreen(name="main"))
        sm.add_widget(ElbowScreen(name="elbow"))
        sm.add_widget(KneeScreen(name="knee"))
        sm.add_widget(ShoulderScreen(name="shoulder"))

        # ست کردن متن فارسی دکمه‌ها و برگشت
        sm.get_screen("main").ids.btn_elbow.text = fix_farsi("آرنج")
        sm.get_screen("main").ids.btn_knee.text = fix_farsi("زانو")
        sm.get_screen("main").ids.btn_shoulder.text = fix_farsi("شانه")
        sm.get_screen("elbow").ids.btn_back_elbow.text = fix_farsi("بازگشت")
        sm.get_screen("knee").ids.btn_back_knee.text = fix_farsi("بازگشت")
        sm.get_screen("shoulder").ids.btn_back_shoulder.text = fix_farsi("بازگشت")

        return sm

    def on_stop(self):
        cap.release()

if __name__=="__main__":
    MyApp().run()
