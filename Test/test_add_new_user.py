import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestUsernameValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = webdriver.Chrome()
        cls.driver.get("http://127.0.0.1:5000")  # Flask uygulamasını başlatmayı unutmayın!

    @classmethod
    def tearDownClass(cls):
        cls.driver.quit()

    def test_turkish_characters_in_username(self):
        driver = self.driver

        # Türkçe karakter içeren kullanıcı adı gir
        username_input = driver.find_element(By.ID, "username")
        username_input.clear()
        username_input.send_keys("TürkçeAdı")

        # "Kullanıcı Ekle" butonuna tıkla
        add_user_button = driver.find_element(By.ID, "addUserButton")
        add_user_button.click()

        # Hata mesajını kontrol et
        try:
            error_message = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "message"))
            )
            print("Hata Mesajı:", error_message.text)
            self.assertIn(
                "türkçe karakterler içermemelidir",
                error_message.text.lower(),
                "Türkçe karakterler için bir hata mesajı bekleniyor."
            )
        except Exception as e:
            print("Hata mesajı alınamadı: ", str(e))
            self.fail("Türkçe karakterlerin kontrolü sırasında hata oluştu.")

if __name__ == "__main__":
    unittest.main()
