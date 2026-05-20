import cv2
import mediapipe as mp
import pyttsx3
import queue
import threading


TIP_INDICES = (4, 8, 12, 16, 20)
THUMB_INDEX_PINCH = 0.05 ** 2
FAR_FINGER_DISTANCE = 0.10 ** 2
THUMB_MIDDLE_PINCH = 0.05 ** 2
INDEX_MIDDLE_PINCH = 0.04 ** 2
COOLDOWN_FRAMES = 20
WINDOW_NAME = "Hand Gesture Recognition"


def squared_distance(point_a, point_b):
    dx = point_a.x - point_b.x
    dy = point_a.y - point_b.y
    return (dx * dx) + (dy * dy)


def gesture_to_text(landmarks):
    if landmarks is None:
        return None

    thumb_tip, index_tip, middle_tip, _, _ = (landmarks[index] for index in TIP_INDICES)

    dist_thumb_index = squared_distance(thumb_tip, index_tip)
    dist_thumb_middle = squared_distance(thumb_tip, middle_tip)
    dist_index_middle = squared_distance(index_tip, middle_tip)

    if dist_thumb_index < THUMB_INDEX_PINCH and dist_thumb_middle > FAR_FINGER_DISTANCE:
        return "Hello"
    if dist_thumb_index < THUMB_INDEX_PINCH and dist_thumb_middle < THUMB_MIDDLE_PINCH:
        return "How are you"
    if dist_index_middle < INDEX_MIDDLE_PINCH and dist_thumb_index > FAR_FINGER_DISTANCE:
        return "Nice to meet you"
    return None


class SpeechWorker:
    def __init__(self):
        self._queue = queue.Queue(maxsize=1)
        self._stop_signal = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text):
        if self._queue.full():
            return False

        self._queue.put_nowait(text)
        return True

    def stop(self):
        try:
            self._queue.put_nowait(self._stop_signal)
        except queue.Full:
            self._queue.get_nowait()
            self._queue.put_nowait(self._stop_signal)
        self._thread.join(timeout=1)

    def _run(self):
        engine = pyttsx3.init()
        while True:
            item = self._queue.get()
            if item is self._stop_signal:
                break

            engine.say(item)
            engine.runAndWait()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    speech_worker = SpeechWorker()
    cooldown_counter = 0
    recent_text = ""

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                result = hands.process(rgb_frame)

                hand_landmarks = None
                if result.multi_hand_landmarks:
                    first_hand = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(frame, first_hand, mp_hands.HAND_CONNECTIONS)
                    hand_landmarks = first_hand.landmark

                text = gesture_to_text(hand_landmarks)

                if cooldown_counter > 0:
                    cooldown_counter -= 1

                if text and text != recent_text and cooldown_counter == 0:
                    if speech_worker.speak(text):
                        print(f"Recognized and speaking: {text}")
                        recent_text = text
                        cooldown_counter = COOLDOWN_FRAMES
                elif text is None and cooldown_counter == 0:
                    recent_text = ""

                if text:
                    cv2.putText(
                        frame,
                        text,
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        speech_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
