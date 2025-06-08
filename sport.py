# 導入所需的套件
import requests
import certifi
import os
import webbrowser
import traceback
# import pytesseract # Removed
from PIL import Image
import io
import base64
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
import time
from datetime import datetime
# import numpy as np # Removed
# import cv2 # Removed
# from multiprocessing import Process, Queue, freeze_support # Removed
import json
import urllib3
from dotenv import load_dotenv # 新增
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError # 新增

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 載入 .env 檔案中的環境變數
load_dotenv()

# 從環境變數中讀取 API 金鑰和登入憑證
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ACCOUNT_LOGIN = os.getenv('ACCOUNT')
PASSWORD_LOGIN = os.getenv('PASSWORD')

# ------------------------------------------------------------------------------
# OCR 系統提示 (可在此修改)
# ------------------------------------------------------------------------------
SYSTEM_PROMPT_FOR_OCR = """你是一個圖片文字辨識工具，將接收傳進來的base64，並且取出裡面的文字，不會有任何空格，以json格式回覆：
{
"respond": "text_from_image_recognition"
}"""
# ------------------------------------------------------------------------------

# 檢查 API 金鑰是否存在
if not GEMINI_API_KEY:
    print("警告：GEMINI_API_KEY 未在 .env 檔案中設定或未載入。新的 Gemini 功能可能無法使用。")

# 檢查登入憑證是否存在
if not ACCOUNT_LOGIN or not PASSWORD_LOGIN:
    print("警告：ACCOUNT 或 PASSWORD 未在 .env 檔案中設定或未載入。登入將會失敗。")
    # 可以選擇在這裡 exit() 或讓後續邏輯處理 None 值
    # exit("錯誤：缺少登入憑證。請檢查 .env 檔案。")


# 初始化 OpenAI client
# 如果 GEMINI_API_KEY 未設定，client 仍然會被初始化，但在呼叫時會失敗
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_text_from_image_gemini(base64_image_data: str,
                               model_name: str = "gemini-2.5-flash-preview-05-20"):
    """
    使用 Gemini API (透過 OpenAI 函式庫) 從 Base64 圖片資料中提取文字。

    Args:
        base64_image_data (str): Base64 編碼的圖片資料 (包含 data URI 前綴，例如 "data:image/jpeg;base64,...").
        model_name (str, optional): 要使用的模型名稱。
                                    預設為 "gemini-2.5-flash-preview-05-20"。

    Returns:
        str: 辨識出的文字，如果成功。
             如果失敗或 AI 回應格式不符，則回傳 None。
    """
    if not GEMINI_API_KEY:
        print("[!] Gemini API 金鑰未設定。請檢查 .env 檔案。")
        return None

    if not base64_image_data:
        print("[!] 未提供 Base64 圖片資料。")
        return None

    # 確保 base64_image_data 包含 data URI scheme
    # 如果 get_captcha 回傳的 base64 不含前綴，則在這裡加上
    # 假設圖片是 PNG 格式，如果不是，需要調整 mime type
    if not base64_image_data.startswith('data:image'):
        base64_image_data = f"data:image/png;base64,{base64_image_data}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_OCR},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "請辨識這張圖片中的文字，並嚴格按照系統提示的JSON格式回覆。"},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image_data} # Base64 data URI
                },
            ],
        },
    ]

    try:
        print(f"[*] 正在使用 OpenAI 函式庫向 Gemini API (模型: {model_name}) 發送圖片辨識請求...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            # max_tokens=150 
        )
        
        print(f"[*] API 請求成功。")
        
        if response.choices and len(response.choices) > 0:
            ai_message = response.choices[0].message
            if ai_message and ai_message.content:
                ai_response_content = ai_message.content
                print(f"[*] AI 原始回覆內容: {ai_response_content}")
                try:
                    cleaned_content = ai_response_content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[len("```json"):].strip()
                    if cleaned_content.endswith("```"):
                        cleaned_content = cleaned_content[:-len("```")].strip()
                    
                    parsed_json = json.loads(cleaned_content)
                    extracted_text = parsed_json.get("respond")
                    if extracted_text is not None:
                        return extracted_text
                    else:
                        print("[!] AI 回應的 JSON 中未找到 \'respond\' 欄位。")
                        print(f"    解析後的 JSON: {parsed_json}")
                        return None
                except json.JSONDecodeError:
                    print("[!] AI 回應不是有效的 JSON 格式，或清理後仍無法解析。")
                    print(f"    清理前的 AI 回應內容: {ai_response_content}")
                    return None
            else:
                print("[!] API 回應中未找到有效的訊息內容。")
                print(f"    完整回應: {response}")
                return None
        else:
            print("[!] API 回應中未找到 \'choices\'。")
            print(f"    完整回應: {response}")
            return None

    except APITimeoutError:
        print(f"[!] 請求超時。")
        return None
    except APIConnectionError as conn_err:
        print(f"[!] 連線錯誤: {conn_err}")
        return None
    except APIError as api_err:
        print(f"[!] Gemini API 錯誤 (狀態碼: {api_err.status_code if hasattr(api_err, 'status_code') else 'N/A'})")
        error_body = api_err.body if hasattr(api_err, 'body') else {}
        error_message = error_body.get('error', {}).get('message', str(api_err)) if isinstance(error_body, dict) else str(api_err)
        print(f"    錯誤訊息: {error_message}")
        return None
    except Exception as e:
        print(f"[!] 呼叫 Gemini API 進行圖片辨識時發生未預期錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_captcha(driver):
    """
    從目前頁面中 ID 為 "imgCaptcha" 的 <img> 標籤獲取 Base64 編碼的驗證碼圖片。

    Args:
        driver: Selenium WebDriver 實例。

    Returns:
        tuple: 包含 (imageBase64_data_uri, None, None) 的元組，如果成功。
               imageBase64_data_uri 是完整的 "data:image/...;base64,..." 字串。
               如果失敗，則回傳 (None, None, None)。
    """
    try:
        print(f"[*] 正在從頁面獲取驗證碼圖片 (ID: imgCaptcha)...")
        # 等待驗證碼圖片元素出現，確保 src 屬性已加載
        captcha_img_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "imgCaptcha"))
        )
        # 增加一個短暫的延遲，確保 src 屬性完全加載 (有時 src 是動態設置的)
        time.sleep(0.5) # 可選，根據實際情況調整
        img_base64_from_page = captcha_img_element.get_attribute("src")

        if img_base64_from_page and img_base64_from_page.startswith('data:image'):
            print(f"[*] 成功從頁面獲取 Base64 圖片資料。")
            # The captchaId and response_cookies are not applicable here
            return img_base64_from_page, None, None
        else:
            print("[!] 從頁面獲取的 src 不是有效的 Base64 data URI 或為空。")
            print(f"    src 屬性內容 (前100字元): {img_base64_from_page[:100] if img_base64_from_page else 'N/A'}...")
            return None, None, None
    except Exception as e:
        print(f"[!] 從頁面獲取驗證碼時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# 設定登入資訊
login_url = "https://sys.ndhu.edu.tw/gc/sportcenter/SportsFields/login.aspx"  # 登入網址
# account = "411122051"    # 帳號 (Moved to .env as ACCOUNT_LOGIN)
# password = "2003.11.05"  # 密碼 (Moved to .env as PASSWORD_LOGIN)

# 設定Chrome瀏覽器選項
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--start-maximized")           # 最大化視窗
chrome_options.add_experimental_option("detach", True)     # 保持瀏覽器開啟
chrome_options.add_argument("--disable-popup-blocking")    # 禁用彈出視窗阻擋

# 啟動瀏覽器
driver = webdriver.Chrome(options=chrome_options)
wait = WebDriverWait(driver, 20)  # 設定等待時間最多20秒

def main_logic(): # 將主要邏輯封裝到一個函數中
    try:
        # 1. 登入流程
        if not ACCOUNT_LOGIN or not PASSWORD_LOGIN:
            print("❌ 錯誤：無法從 .env 檔案讀取帳號或密碼。請檢查 .env 設定。")
            return # 終止函數執行

        driver.get(login_url)     # 開啟登入頁面
        
        # 輸入帳號密碼
        account_input = wait.until(EC.presence_of_element_located((By.ID, "MainContent_TxtUSERNO")))
        password_input = wait.until(EC.presence_of_element_located((By.ID, "MainContent_TxtPWD")))
        account_input.clear()
        password_input.clear()
        account_input.send_keys(ACCOUNT_LOGIN) # Use ACCOUNT_LOGIN from .env
        password_input.send_keys(PASSWORD_LOGIN) # Use PASSWORD_LOGIN from .env
        
        # 點擊登入按鈕
        login_button = wait.until(EC.element_to_be_clickable((By.ID, "MainContent_Button1")))
        login_button.click()
        
        # 2. 點擊進入預約頁面
        button2 = wait.until(EC.element_to_be_clickable((By.ID, "MainContent_Button2")))
        button2.click()
        time.sleep(3)
        
        # 3. 設定預約日期
        try:
            target_date = "2025/06/20"  # 設定目標日期
            
            # 使用JavaScript設定日期
            js_code = f"""
                document.getElementById('MainContent_TextBox1').value = '{target_date}';
                __doPostBack('ctl00$MainContent$TextBox1','');
            """
            driver.execute_script(js_code)
            time.sleep(3)
            
            # 點擊查詢按鈕
            query_button = wait.until(EC.element_to_be_clickable((By.ID, "MainContent_Button1")))
            driver.execute_script("arguments[0].click();", query_button)
            time.sleep(3)
            
        except Exception as e:
            print(f"設定日期時出錯: {e}")
            print(traceback.format_exc())
        
        # 4. 選擇場地
        try:
            court_select = driver.find_element(By.ID, "MainContent_DropDownList1")
            select = Select(court_select)
            
            # 選擇特定場地 
            court_value = "VOL0A"
            select.select_by_value(court_value)
            time.sleep(1)
            # 觸發場地選擇的更新
            driver.execute_script("__doPostBack('ctl00$MainContent$DropDownList1','')")
            time.sleep(2)
            
        except Exception as e:
            print(f"選擇場地時出錯: {e}")
            print(traceback.format_exc())
        
        # 5. 選擇時段
        try:
            time.sleep(2)
            
            # 定義預約時段
            desired_time = "06~08"
            
            # 修改後的XPath，更精確地定位按鈕
            time_slot_xpath = f"""
                //tr[
                    td[contains(text(), '06')] and 
                    td/button[contains(@type, 'button') and 
                    contains(., '[申請]') and 
                    contains(., '{desired_time}')]
                ]//button
            """
            
            # 等待按鈕出現並點擊
            time_slot_button = wait.until(
                EC.presence_of_element_located((By.XPATH, time_slot_xpath))
            )
            
            # 使用JavaScript點擊按鈕
            driver.execute_script("arguments[0].scrollIntoView(true);", time_slot_button)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", time_slot_button)
            time.sleep(2)

        except Exception as e:
            print(f"選擇時段時出錯: {e}")
            print(traceback.format_exc())
            
        # 6. 處理驗證碼
        try:
            attempt_count = 0
            success_count = 0
            failure_count = 0
            max_retries = 5
            
            debug_image_folder = "captcha_debug_images"
            if not os.path.exists(debug_image_folder):
                os.makedirs(debug_image_folder)
            
            if not GEMINI_API_KEY:
                print("‼️ 警告: 未能載入 Gemini API 金鑰 (從 .env)。Gemini API 將無法使用。")
            else:
                print("🔑 Gemini API 金鑰已成功從 .env 載入。")

            while attempt_count < max_retries:
                try:
                    attempt_count += 1
                    print(f"\n--- 第 {attempt_count} 次嘗試 ---")
                    time.sleep(1) 

                    wait_long = WebDriverWait(driver, 30)

                    print("🔄 正在從當前頁面獲取驗證碼圖片...")
                    img_base64_from_page, _, _ = get_captcha(driver) 
                    
                    if not img_base64_from_page:
                        print(f"⚠️ 從頁面獲取驗證碼圖片失敗 (第 {attempt_count} 次)。")
                        failure_count += 1
                        # Try to click the refresh button if image not found or other issues
                        try:
                            print("🔄 嘗試點擊 '換一張' 按鈕刷新驗證碼 (因獲取圖片失敗)...")
                            refresh_captcha_button_xpath = "//button[@type='button' and @onclick='refreshCaptcha()' and contains(text(),'換一張')]"
                            refresh_button = wait_long.until(EC.element_to_be_clickable((By.XPATH, refresh_captcha_button_xpath)))
                            refresh_button.click()
                            print("ℹ️ 已點擊 '換一張' 按鈕。")
                            time.sleep(2) # Wait for new captcha to load
                        except Exception as e_refresh:
                            print(f"⚠️ 嘗試點擊 '換一張' 按鈕失敗: {e_refresh}")
                        time.sleep(2) 
                        continue 
                    
                    actual_base64_content_for_ocr = ""
                    if ',' in img_base64_from_page: 
                        actual_base64_content_for_ocr = img_base64_from_page.split(',', 1)[1]
                    else:
                        print(f"⚠️ 獲取的 Base64 資料 ({img_base64_from_page[:30]}...) 不包含 'data:image/...;base64,' 前綴，直接使用。")
                        actual_base64_content_for_ocr = img_base64_from_page
                    
                    if img_base64_from_page: # Save if we got something
                        try:
                            base64_content_to_decode = actual_base64_content_for_ocr
                            img_data_for_save = base64.b64decode(base64_content_to_decode)
                            img_for_save = Image.open(io.BytesIO(img_data_for_save))
                            decoded_page_original_path = os.path.join(debug_image_folder, f"page_original_attempt_{attempt_count}.png")
                            img_for_save.save(decoded_page_original_path)
                            print(f"🖼️ 從頁面獲取的原始圖片已儲存至: {decoded_page_original_path}")
                        except Exception as e_save_page_img:
                            print(f"⚠️ 儲存從頁面獲取的原始圖片失敗: {e_save_page_img}")

                    try: 
                        final_captcha_text = ""
                        if GEMINI_API_KEY:
                            print("🧠 正在使用 Gemini API 進行驗證碼辨識...")
                            gemini_result_raw = get_text_from_image_gemini(img_base64_from_page) 
                            
                            if gemini_result_raw is not None: 
                                gemini_result_cleaned = re.sub(r'[^A-Za-z0-9]', '', gemini_result_raw).strip()
                                print(f"🤖 Gemini API 辨識結果 (原始: '{gemini_result_raw}', 清理後: '{gemini_result_cleaned}')")
                                final_captcha_text = gemini_result_cleaned
                            else:
                                print("❌ Gemini API 未返回有效結果或解析失敗。")
                        else:
                            print("⚠️ Gemini API 金鑰未設定，無法進行 OCR。請設定 .env 檔案中的 GEMINI_API_KEY。")
                            failure_count += 1
                            time.sleep(2)
                            continue

                        print(f"📝 最終用於輸入的驗證碼：'{final_captcha_text}' (長度: {len(final_captcha_text)})")

                        if not final_captcha_text or len(final_captcha_text) < 4: 
                            print(f"⚠️ 驗證碼 ('{final_captcha_text}') 長度不足或為空，將刷新並重試。")
                            failure_count += 1
                            try:
                                print("🔄 正在嘗試點擊 '換一張' 按鈕刷新驗證碼 (因辨識結果不足)...")
                                refresh_captcha_button_xpath = "//button[@type='button' and @onclick='refreshCaptcha()' and contains(text(),'換一張')]"
                                refresh_button = wait_long.until(EC.element_to_be_clickable((By.XPATH, refresh_captcha_button_xpath)))
                                refresh_button.click()
                                print("ℹ️ 已點擊 '換一張' 按鈕。")
                                time.sleep(2) 
                            except Exception as e_refresh:
                                print(f"⚠️ 點擊 '換一張' 按鈕失敗: {e_refresh}")
                                print("ℹ️ 將在下次迴圈自動獲取新驗證碼 (如果頁面自動刷新)。")
                            continue 

                        captcha_input_element = wait_long.until(EC.presence_of_element_located((By.ID, "txtCaptchaValue")))
                        captcha_input_element.clear()
                        captcha_input_element.send_keys(final_captcha_text)
                        
                        submit_button_xpath = "//button[@type='button' and text()='申請' and starts-with(@onclick, 'doApp(')]"
                        submit_button_element = wait_long.until(EC.element_to_be_clickable((By.XPATH, submit_button_xpath)))
                        submit_button_element.click()
                        time.sleep(3) 

                        if "申請成功" in driver.page_source or "成功" in driver.page_source:
                            print("🎉 預約成功！")
                            success_count += 1
                            break 
                        else:
                            print("⚠️ 預約失敗，可能為驗證碼錯誤或提交問題。")
                            failure_count += 1
                            # Attempt to refresh captcha if submission failed
                            try:
                                print("🔄 嘗試點擊 '換一張' 按鈕刷新驗證碼 (因提交失敗)...")
                                refresh_captcha_button_xpath = "//button[@type='button' and @onclick='refreshCaptcha()' and contains(text(),'換一張')]"
                                refresh_button = wait_long.until(EC.element_to_be_clickable((By.XPATH, refresh_captcha_button_xpath)))
                                refresh_button.click()
                                print("ℹ️ 已點擊 '換一張' 按鈕。")
                                time.sleep(2) 
                            except Exception as e_refresh_submit_fail:
                                print(f"⚠️ 點擊 '換一張' 按鈕失敗 (因提交失敗): {e_refresh_submit_fail}")
                            # time.sleep(2) # Already covered by try-except or loop delay
                            # continue will be hit implicitly

                    except Exception as e_attempt: 
                        print(f"🚨 內部嘗試 (第 {attempt_count} 次) 時發生錯誤: {e_attempt}")
                        print(traceback.format_exc())
                        failure_count += 1
                        if attempt_count < max_retries: 
                            print("ℹ️ 因內部錯誤，下次迴圈將自動獲取新驗證碼。")
                            time.sleep(2) 
                
                except Exception as e_loop:
                    print(f"❌ 處理驗證碼迴圈內部出錯: {e_loop}")
                    print(traceback.format_exc())
                    if attempt_count >= max_retries:
                        print("達到最大重試次數，程式終止")
                        break
                    time.sleep(3)

            print(f"✅ 成功次數: {success_count}, ❌ 失敗次數: {failure_count}")
            print("🎯 驗證碼處理結束")

        except Exception as e_captcha_module:
            print(f"驗證碼模組發生嚴重錯誤: {e_captcha_module}")
            print(traceback.format_exc())

    finally:
        print("程式執行完畢。如果瀏覽器保持開啟是預期行為，請忽略此訊息。")
        # driver.quit() # 如果需要在結束時關閉瀏覽器，取消此行註解

if __name__ == '__main__':
    # freeze_support() # Removed as multiprocessing is no longer used
    main_logic() # 呼叫主邏輯函數
