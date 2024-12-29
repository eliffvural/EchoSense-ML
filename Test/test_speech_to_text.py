import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestSpeechToText(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Chrome()
        cls.driver.get("http://127.0.0.1:5000")  # Flask uygulamasını başlatmayı unutmayın!

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_speech_to_text_conversion(self):
        driver = self.driver

        # "Konuşmaya Başla" butonunu bul ve tıkla
        start_speech_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Konuşmaya Başla')]")
        start_speech_button.click()

        # Anlık metnin DOM'da güncellenmesini kontrol et
        try:
            transcription_box = WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.ID, "transcription"))
            )
            print("Transcription Box Text:", transcription_box.text)
            self.assertNotEqual(transcription_box.text.strip(), "", "Transcription box should not be empty.")
        except Exception as e:
            print("Metin kutusu güncellenmedi: ", str(e))
            self.fail("Sesin metne dönüştürülmesi başarısız.")

if __name__ == "__main__":
    unittest.main()
