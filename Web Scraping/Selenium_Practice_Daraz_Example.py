#seaborn example code line code 137
#sorted_df = dffilter.sort_values(by='date_added')
"""g=sns.displot(data=sorted_df, x="price" , y="date_added" , kind='kde'  )
g.figure.suptitle("sns.displot(data=dffilter, x=price , y=date_added , kind='kde'  )"  )"""


# Import Selenium WebDriver module
# Selenium controls the browser automatically
from selenium import webdriver 

# Import keyboard key functions like ENTER, ESCAPE etc.
from selenium.webdriver.common.keys import Keys

# Import By class
# Used to locate HTML elements on webpage
from selenium.webdriver.common.by import By

# Import csv module
# Used to save data into CSV file
import csv


# URL of Daraz smartphones page
url = "https://www.daraz.pk/catalog/?spm=a2a0e.tm80331704.cate_5.5.77cc5aa7fPImi7&q=Smart%20Phones&from=hp_categories&src=all_channel"


# Create ChromeService object
# executable_path tells Selenium where chromedriver.exe is located
cService = webdriver.ChromeService(
    executable_path='C:\\Users\\pc\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe'
)


# Launch Chrome browser using chromedriver
driver = webdriver.Chrome(service=cService)


# Open the Daraz URL in browser
driver.get(url) 


# Print message in terminal
print("Scrolling to load products...")


# Loop runs 5 times
# Used to scroll webpage down slowly
for i in range(5):

    # Execute JavaScript inside browser
    # Scroll down webpage by 800 pixels
    driver.execute_script("window.scrollBy(0, 800);")


# Find all product containers using class name
# Each product card has class "Bm3ON"
productsdiv = driver.find_elements(By.CLASS_NAME, "Bm3ON")


# Create empty list
# This list will store all product data
products=[]


# Print total products found
print(f"Found {len(productsdiv)} products. Extracting URLs...")


# Loop through each product card
for product in productsdiv:


    # Find anchor <a> tag inside product
    # <a> tag contains product link
    name = product.find_element(By.TAG_NAME, "a")


    # Get href attribute from anchor tag
    # href contains product URL
    url =  name.get_attribute('href')


    # Remove unnecessary tracking parameters from URL
    # Example:
    # before = product.com?id=123&spm=abc
    # after  = product.com?id=123
    url = url.split("?")[0] if url else "N/A"


    # Print product URL
    print(url)


    # Find product price using class name "ooOxS"
    p_price = product.find_element(By.CLASS_NAME, "ooOxS").text


    # Split text and keep only price value
    # Example:
    # "Rs. 45,000" → "45,000"
    p_price = p_price.split()[1] if p_price else "N/A"


    # Print product price
    print(p_price)

    # Find product name using class name "RfADt"
    p_name = product.find_element(By.CLASS_NAME, "RfADt").text


    # Split text and keep first word only
    # Example:
    # "Samsung Galaxy A15" → "Samsung"
    p_name = p_name.split()[0] if p_name else "N/A"


    # Print product name
    print(p_name)


    # Find units sold using class name "_6uN7R"
    units_sold = product.find_element(By.CLASS_NAME, "_6uN7R").text


    # Keep only first word
    # Example:
    # "12 Sold" → "12"
    units_sold = units_sold.split()[0] if units_sold else "N/A"


    # Print units sold
    print(units_sold)
    

    # Add extracted data into products list as dictionary
    products.append({

        # Dictionary key-value pairs
        'Product Name': p_name,
        'Price': p_price,
        'Product URL': url,
        'Units Sold': units_sold
    })


# Open CSV file in write mode
# newline='' prevents empty rows
# utf-8 supports special characters
with open('Daraz_products_Selenium.csv',
          'w',
          newline='',
          encoding='utf-8') as csvfile:


    # Create CSV writer object
    writer = csv.DictWriter(
        csvfile,
        fieldnames=[
            'Product Name',
            'Price',
            'Product URL',
            'Units Sold'
        ]
    )


    # Write column headings in CSV
    writer.writeheader()


    # Write all product rows into CSV
    writer.writerows(products)  


# Print success message
print("Data saved to Daraz_products_Selenium.csv")


# Close browser completely
driver.quit()