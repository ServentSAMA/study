from playwright.sync_api import sync_playwright


def test_has_title():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("D:\pythonProject\study\完美世界207集磁链.html")
        # 选择所有 <a> 标签
        a_elements = page.locator("a").all()
        print()
        for a in a_elements:
            # 获取 value 属性（如果存在）
            value = a.get_attribute("href")

            # 判断字符串是否包含 magnet
            if value and "magnet" in value:
                print(value)

        browser.close()