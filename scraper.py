from __future__ import annotations
import time, re
from typing import List, Dict

from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

import config as C


# Driver setup
def make_driver(headless: bool = False):
    opts = Options()
    if headless:
        
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1600,1000")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--no-sandbox")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


# Helper functions
def wait_for_list(driver: webdriver.Chrome, wait: WebDriverWait):
    wait.until(EC.presence_of_element_located((By.ID, "__xmlview1--jobResultList")))
    wait.until(EC.presence_of_all_elements_located((By.XPATH, "//li[contains(@class,'sapMLIBTypeNavigation')]")))

def _get_trigger_counter_text(driver: webdriver.Chrome):
    try:
        el = driver.find_element(By.ID, "__xmlview1--jobResultList-triggerInfo")
        return el.text.strip()
    except Exception:
        return ""

def _parse_counter_n_shown(counter_text: str):
    # Formats like: [ 250 / 377 ]
    m = re.search(r"\[\s*(\d+)\s*/", counter_text)
    return int(m.group(1)) if m else -1

def _click_growing_trigger(driver: webdriver.Chrome):     #clicks the 'Mehr anzeigen' growing-list trigger with multiple fallbacks.
  
    targets: List[tuple[str, str]] = [
        ("id", "__xmlview1--jobResultList-trigger"),
        ("css", "#__xmlview1--jobResultList-trigger-content"),
        ("id", "__xmlview1--jobResultList-triggerText"),
        ("xpath", "//*[contains(@class,'sapMSLIDiv') and .//span[@id='__xmlview1--jobResultList-triggerText']]"),
    ]

    for how, sel in targets:
        try:
            if how == "id":
                el = driver.find_element(By.ID, sel)
            elif how == "css":
                el = driver.find_element(By.CSS_SELECTOR, sel)
            else:
                el = driver.find_element(By.XPATH, sel)

            try:
                clickable = el.find_element(By.XPATH, "./ancestor::*[self::li or self::div][1]")
            except Exception:
                clickable = el

            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", clickable)
            time.sleep(0.15)

            
            try:
                clickable.click() #try normal click
                return True
            except Exception:
                pass

            
            try:
                driver.execute_script("arguments[0].click();", clickable) #JS click
                return True
            except Exception:
                pass

            # Keyboard activation
            try:
                clickable.send_keys(Keys.ENTER)
                return True
            except Exception:
                pass

            # Dispatch mouse events
            try:
                driver.execute_script("""
                  const el = arguments[0];
                  ['mousedown','mouseup','click'].forEach(t => {
                    el.dispatchEvent(new MouseEvent(t, {bubbles:true,cancelable:true,view:window}));
                  });
                """, clickable)
                return True
            except Exception:
                pass

        except Exception:
            continue

    return False

def load_all_jobs(driver: webdriver.Chrome, wait: WebDriverWait):    #clicks the growing-list trigger until no more items are added.

    last_rows = len(driver.find_elements(By.XPATH, "//li[contains(@class,'sapMLIBTypeNavigation')]"))
    last_counter_text = _get_trigger_counter_text(driver)
    last_shown = _parse_counter_n_shown(last_counter_text)

    stagnant_cycles = 0

    with tqdm(desc="Loading job cards", unit="jobs") as pbar:
        pbar.update(last_rows)

        while True:
            clicked = _click_growing_trigger(driver)

            if not clicked:
                stagnant_cycles += 1
                if stagnant_cycles >= 3:
                    pbar.write("End of list (trigger not clickable).")
                    break
                driver.execute_script("window.scrollBy(0, 900);")
                time.sleep(0.5)
                continue

            # After click, allow UI to fetch & render
            time.sleep(0.5)

            grew = False
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: len(d.find_elements(By.XPATH, "//li[contains(@class,'sapMLIBTypeNavigation')]")) > last_rows
                )
                grew = True
            except Exception:
                for _ in range(20):
                    new_counter_text = _get_trigger_counter_text(driver)
                    new_shown = _parse_counter_n_shown(new_counter_text)
                    if new_shown != -1 and new_shown > last_shown:
                        grew = True
                        break
                    time.sleep(0.25)

            new_rows = len(driver.find_elements(By.XPATH, "//li[contains(@class,'sapMLIBTypeNavigation')]"))
            new_counter_text = _get_trigger_counter_text(driver)
            new_shown = _parse_counter_n_shown(new_counter_text)

            if grew and new_rows > last_rows:
                pbar.update(new_rows - last_rows)
                last_rows = new_rows
                last_counter_text = new_counter_text
                last_shown = new_shown if new_shown != -1 else last_shown
                stagnant_cycles = 0
                time.sleep(0.2)
                continue

            stagnant_cycles += 1
            if stagnant_cycles >= 3:
                pbar.write("End of list reached (no growth).")
                break

            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(0.5)

    return last_rows

def parse_list_item_fields(li_html: str):
    
    soup = BeautifulSoup(li_html or "", "html.parser")
    title_el = soup.find("h3")
    title = title_el.get_text(strip=True) if title_el else ""
    dl = soup.find("span", string=lambda s: s and "Bewerbungsende:" in s)
    deadline = (dl.get_text(strip=True).split(":", 1)[-1].strip()) if dl else ""
    loc = soup.find("span", string=lambda s: s and "Dienstort:" in s)
    location = (loc.get_text(strip=True).split(":", 1)[-1].strip()) if loc else ""
    
    return {"title": title, "deadline": deadline, "location": location}

def click_row_by_index(driver: webdriver.Chrome, wait: WebDriverWait, idx1: int):     #click row i (1-based) using nav chevron first, then LI fallback.

    try:
        chevron = driver.find_element(
            By.XPATH,
            f"(//span[contains(@id,'-imgNav') and contains(@class,'sapMLIBImgNav')])[{idx1}]"
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", chevron)
        time.sleep(C.SCRAPER_AFTER_CLICK_DELAY)
        chevron.click()
        return True
    except Exception:
        pass

    try:
        li = driver.find_element(By.XPATH, f"(//li[contains(@class,'sapMLIBTypeNavigation')])[{idx1}]")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", li)
        time.sleep(C.SCRAPER_AFTER_CLICK_DELAY)
        ActionChains(driver).move_to_element(li).click().perform()
        return True
    except Exception:
        pass

    try:
        li = driver.find_element(By.XPATH, f"(//li[contains(@class,'sapMLIBTypeNavigation')])[{idx1}]")
        li.send_keys(Keys.SPACE)
        time.sleep(0.2)
        li.send_keys(Keys.ENTER)
        return True
    except Exception:
        return False


# Detail helpers
def wait_for_detail(driver: webdriver.Chrome) -> None:
    signals = [
        (By.ID, "jobDetailsViewSkipTarget"),
        (By.XPATH, "//*[contains(@id,'jobDetailsTasksAccordionSection-expandButton')]"),
        (By.XPATH, "//button[contains(.,'Druckvorschau') or contains(.,'Online bewerben')]"),
    ]
    last_err = None
    for _ in range(C.SCRAPER_WAIT * 2):
        for how, what in signals:
            try:
                if driver.find_elements(how, what):
                    return
            except Exception as e:
                last_err = e
        time.sleep(0.4)
    raise TimeoutException(str(last_err) if last_err else "detail signals not found")

def expand_panel_and_extract(driver: webdriver.Chrome, wait: WebDriverWait,
                             panel_suffix_key: str, heading_text_fallback: str) -> List[str]:
  
    try:
        expand_btn = wait.until(EC.presence_of_element_located(
            (By.XPATH, f"//button[contains(@id,'--{panel_suffix_key}-expandButton')]")
        ))
    except TimeoutException:
        title_span = wait.until(EC.presence_of_element_located(
            (By.XPATH, f"//h2//span[normalize-space()='{heading_text_fallback}']"))
        )
        panel = title_span.find_element(By.XPATH, "./ancestor::div[contains(@class,'sapMPanel')]")
        expand_btn = panel.find_element(By.XPATH, ".//button[contains(@id,'-expandButton')]")

    driver.execute_script("arguments[0].scrollIntoView({block:'center'});", expand_btn)
    time.sleep(0.2)
    aria = (expand_btn.get_attribute("aria-expanded") or "").lower()
    if aria == "false":
        try:
            wait.until(EC.element_to_be_clickable(expand_btn))
            expand_btn.click()
        except Exception:
            driver.execute_script("arguments[0].click();", expand_btn)

    try:
        content = wait.until(EC.visibility_of_element_located(
            (By.XPATH, f"//*[contains(@id,'--{panel_suffix_key}-content')]")
        ))
    except TimeoutException:
        content = wait.until(EC.visibility_of_element_located(
            (By.XPATH, f"//*[contains(@id,'{panel_suffix_key}-content')]")
        ))

    # Give UI time to render 
    time.sleep(C.SCRAPER_RENDER_DELAY)

    end_time = time.time() + 8
    while time.time() < end_time:
        try:
            lis = content.find_elements(By.CSS_SELECTOR, "li")
            if lis and any(li.text.strip() for li in lis):
                break
            inner = content.get_dom_property("innerText") or ""
            if inner.strip():
                break
        except Exception:
            pass
        time.sleep(0.2)

    lines: List[str] = []
    try:
        for li in content.find_elements(By.CSS_SELECTOR, "li"):
            txt = li.text.strip()
            if txt:
                lines.append(txt)
    except Exception:
        pass

    if not lines:
        try:
            raw = content.get_dom_property("innerText") or ""
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
        except Exception:
            pass

    if not lines:
        html = content.get_attribute("innerHTML") or ""
        soup = BeautifulSoup(html, "html.parser")
        li_texts = [li.get_text(" ", strip=True) for li in soup.find_all("li")]
        if li_texts:
            lines = li_texts
        else:
            for el in soup.find_all(['div', 'p']):
                t = el.get_text(" ", strip=True)
                if t:
                    lines.append(t)

    seen, out = set(), []
    for s in lines:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def main():
    # Ensure output directory exists (data/)
    Path(C.DATA_DIR).mkdir(parents=True, exist_ok=True)

    driver = make_driver(headless=C.SCRAPER_HEADLESS)
    wait = WebDriverWait(driver, C.SCRAPER_WAIT)

    try:
        # 1) Open list, accept cookies (best-effort), load all jobs
        driver.get(C.SCRAPER_URL)

        try:
            btn = WebDriverWait(driver, 8).until(
                EC.presence_of_element_located((By.XPATH, "//button[contains(., 'Akzeptieren') or contains(., 'Accept')]"))
            )
            if btn.is_displayed():
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(0.2)
        except Exception:
            pass

        wait_for_list(driver, wait)
        total = load_all_jobs(driver, wait)
        print(f"Total jobs loaded: {total}")

        results = []

        # 2) Iterate job-by-job
        print("🧠 Scraping detail pages…")
        for idx in tqdm(range(1, total + 1), desc="Jobs", unit="job"):
            try:
                li = driver.find_element(By.XPATH, f"(//li[contains(@class,'sapMLIBTypeNavigation')])[{idx}]")
                li_html = li.get_attribute("innerHTML") or ""
            except Exception:
                driver.get(C.SCRAPER_URL)
                wait_for_list(driver, wait)
                load_all_jobs(driver, wait)
                li = driver.find_element(By.XPATH, f"(//li[contains(@class,'sapMLIBTypeNavigation')])[{idx}]")
                li_html = li.get_attribute("innerHTML") or ""

            basics = parse_list_item_fields(li_html)

            if not click_row_by_index(driver, wait, idx):
                tqdm.write(f"Could not open row {idx}; skipping.")
                continue

            try:
                wait_for_detail(driver)
            except TimeoutException:
                tqdm.write(f"Detail view timeout for row {idx}; skipping.")
                try:
                    driver.back(); wait_for_list(driver, wait)
                except Exception:
                    driver.get(C.SCRAPER_URL); wait_for_list(driver, wait); load_all_jobs(driver, wait)
                continue

            try:
                detail_title = driver.find_element(By.XPATH, "//h1|//h2").text.strip()
                if detail_title:
                    basics["title"] = detail_title
            except Exception:
                pass

            current_url = driver.current_url

            try:
                aufgaben_lines = expand_panel_and_extract(
                    driver, wait,
                    panel_suffix_key="jobDetailsTasksAccordionSection",
                    heading_text_fallback="Aufgaben und Tätigkeiten"
                )
            except Exception:
                aufgaben_lines = []

            try:
                req_lines = expand_panel_and_extract(
                    driver, wait,
                    panel_suffix_key="jobDetailsRequirementsAccordionSection",
                    heading_text_fallback="Erfordernisse"
                )
            except Exception:
                req_lines = []

            results.append({
                "title": basics.get("title", ""),
                "url": current_url,
                "deadline": basics.get("deadline", ""),
                "location": basics.get("location", ""),
                "aufgaben": "\n".join(aufgaben_lines),
                "erfordernisse": "\n".join(req_lines),
            })

            # Go back to list for next item
            try:
                driver.back()
                wait_for_list(driver, wait)
            except Exception:
                driver.get(C.SCRAPER_URL)
                wait_for_list(driver, wait)
                load_all_jobs(driver, wait)

            time.sleep(0.1)

        # 3) Save CSV (relative path from config)
        out_path = Path(C.SCRAPER_CSV_OUT)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results, columns=[
            "title", "url", "deadline", "location", "aufgaben", "erfordernisse_list"
        ]).to_csv(out_path, index=False)
        print(f"Saved {len(results)} rows to {out_path}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
