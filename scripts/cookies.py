from requests.cookies import RequestsCookieJar

COOKIES_LIST = []

# Create a RequestsCookieJar instance
COOKIES = RequestsCookieJar()

# Add cookies to the jar
for cookie in COOKIES_LIST:
    COOKIES.set(cookie["name"], cookie["value"], domain=cookie["domain"], path=cookie["path"])
