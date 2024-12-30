from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import time

# 使用 webdriver-manager 自動下載 ChromeDriver
service = Service(ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument("--headless")  # 無頭模式（背景執行）
driver = webdriver.Chrome(service=service, options=options)

# 目標網址
url = "https://www.taiwanlottery.com/lotto/result/super_lotto638"
driver.get(url)
time.sleep(2)  # 等待網頁加載

# 獲取 HTML
html_content = driver.page_source
soup = BeautifulSoup(html_content, "html.parser")

# 找到威力彩的結果區塊
result_div = soup.find("div", class_="result-item-simple-area-container")
if result_div:
    big_order_numbers = [ball.text.strip() for ball in result_div.find_all("div", class_="ball")]

    # 點擊切換按鈕
    try:
        sort_switch_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "sort-switch-btn"))
        )
        ActionChains(driver).move_to_element(sort_switch_button).click().perform()
        time.sleep(1)

        # 重新抓取資料
        html_content_updated = driver.page_source
        soup_updated = BeautifulSoup(html_content_updated, "html.parser")
        open_order_numbers = [ball.text.strip() for ball in soup_updated.find_all("div", class_="ball")[:7]]

        # 第二區號碼
        secondary_number = result_div.find("div", class_="ball color-super secondary-area")
        secondary_value = secondary_number.get_text(strip=True) if secondary_number else "未找到第二區號碼"

        # 顯示結果
        print("大小順序:", ", ".join(big_order_numbers))
        print("開出順序:", ", ".join(open_order_numbers))
        print("第二區:", secondary_value)
    except Exception as e:
        print("按鈕無法點擊或數據抓取失敗:", e)
else:
    print("未找到威力彩的開獎結果")

# 關閉瀏覽器
driver.quit()
