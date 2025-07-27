import unittest
from solution.app import text_to_audio, audio_to_text
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

class TestShannonCodec(unittest.TestCase):

    def test_simple_encoding_decoding(self):
        original = "3141592653"
        wav_bytes = text_to_audio(original)
        decoded = audio_to_text(wav_bytes)
        print("decoded:", decoded)

        distance = levenshtein_distance(original, decoded)
        similarity = 1 - distance / len(original)
        self.assertGreaterEqual(similarity, 0.9)

    def test_max_duration(self):
        text = "1234567890" * 10  # 100 символов
        wav_bytes = text_to_audio(text)
        self.assertLessEqual(len(wav_bytes) / 88200, 10)

    def test_correct_format(self):
        wav_bytes = text_to_audio("42")
        from wave import open as wave_open
        import io
        with wave_open(io.BytesIO(wav_bytes), 'rb') as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2)  # 16-bit
            self.assertEqual(wf.getframerate(), 44100)

def levenshtein_distance(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[-1][-1]

if __name__ == '__main__':
    unittest.main()