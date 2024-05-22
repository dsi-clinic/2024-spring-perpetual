"""Provides an interface to interact with the web platform Padlet.
"""

# Standard library imports
import logging
import os
import time
from typing import Callable, Dict, List
from urllib.parse import urlparse

# Third-party imports
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class PadletClient:
    """Provides access to create new boards and add and download board posts."""

    BROWSER_TITLE_CHECK_STRING: str = "Padlet"
    """A string used to confirm that the webpage has loaded as expected.
    """

    DEFAULT_SECONDS_WAIT_PAGE_ACTION: float = 2
    """The default number of seconds to wait for a page action to complete.
    """

    DEFAULT_SECONDS_WAIT_PAGE_LOAD: float = 5
    """The default number of seconds to wait for a page load to complete.
    """

    MAX_ATTEMPTS_PAGE_RELOAD: int = 3
    """The maximum number of times to attempt to load a webpage before raising an error.
    """

    SECONDS_DELAY_PER_REQUEST = 60 / 250
    """The number of seconds to wait in between API requests.
    Accommodates the maximum number of requests per minute (250).
    """

    def __init__(self, logger: logging.Logger) -> None:
        """Initializes a new instance of a `PadletClient`.

        Args:
            logger (`logging.Logger`): An instance of a Python
                standard logger.

        Raises:
            `RuntimeError` if one of the environment variables
                `PADLET_HOMEPAGE_URL`, `PADLET_USERNAME`,
                `PADLET_PASSWORD`, `PADLET_TEMPLATE_INDOOR_BINS_SECTION`,
                `PADLET_TEMPLATE_OUTDOOR_BINS_SECTION`, or
                `PADLET_API_KEY` is not found.

        Returns:
            `None`
        """
        # Parse environment variables
        try:
            self._homepage_url = os.environ["PADLET_HOMEPAGE_URL"]
            self._user_name = os.environ["PADLET_USER_NAME"]
            self._password = os.environ["PADLET_PASSWORD"]
            self._api_key = os.environ["PADLET_API_KEY"]
            self._template_name = os.environ["PADLET_TEMPLATE_NAME"]
            self._indoor_bins_section = os.environ[
                "PADLET_TEMPLATE_INDOOR_BINS_SECTION"
            ]
            self._outdoor_bins_section = os.environ[
                "PADLET_TEMPLATE_OUTDOOR_BINS_SECTION"
            ]
        except KeyError as e:
            raise RuntimeError(
                "Failed to initialize PadletClient."
                f'Missing expected environment variable "{e}".'
            ) from None

        # Set remaining fields
        self._browser = None
        self._logger = logger

    def _initialize_browser(self) -> None:
        """Launches a new browser at the Padlet homepage and
        authenticates, updating the current browser class state.

        Args:
            `None`

        Returns:
            `None`
        """
        # Launch browser and authenticate
        self._logger.info(
            f'Launching new web browser at "{self._homepage_url}".'
        )
        self._launch_homepage()

        # Log into site using user name and password
        self._logger.info("Logging in with user name and password.")
        self._authenticate()

        # Log results
        self._logger.info("Browser successfully initialized.")

    def _launch_homepage(self) -> None:
        """Launches a new browser window controlled by
        a Chrome WebDriver to the Padlet homepage and
        saves it to the class state.

        Args:
            `None`

        Returns:
            `None`
        """
        # Initialize attempt counter
        num_attempts = 0

        while num_attempts < self.MAX_ATTEMPTS_PAGE_RELOAD:
            # Initialize default options for headless Chrome WebDriver
            chromeOptions = webdriver.ChromeOptions()
            chromeOptions.add_argument("--headless")
            chromeOptions.add_argument("--disable-dev-shm-usage")
            chromeOptions.add_argument("--hide-scrollbars")
            browser = webdriver.Chrome(options=chromeOptions)

            # Navigate to page and wait to load
            browser.get(self._homepage_url)
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)

            # Confirm page has loaded correctly or raise error if attempts exhausted
            if self.BROWSER_TITLE_CHECK_STRING in browser.title:
                break
            elif num_attempts >= self.MAX_ATTEMPTS_PAGE_RELOAD:
                raise RuntimeError("Page failed to load correctly.")
            else:
                num_attempts += 1

        self._browser = browser

    def _authenticate(self) -> None:
        """Logs into the Padlet Platform from the site homepage,
        therefore changing the browser instance in place.

        Args:
            `None`

        Returns:
            `None`
        """
        # Click the log in button
        try:
            selector = (By.CSS_SELECTOR, "a[data-testid='logInButton']")
            condition = EC.element_to_be_clickable(selector)
            btn = WebDriverWait(self._browser, 5).until(condition)
            btn.click()
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)
        except Exception as e:
            raise RuntimeError(
                f"Failed to log into Padlet platform. {e}"
            ) from None

        # Fill in user name input box on new page
        try:
            selector = (By.ID, "username-input")
            condition = EC.element_to_be_clickable(selector)
            input = WebDriverWait(self._browser, 5).until(condition)
            input.send_keys(self._user_name)
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_ACTION)
        except Exception as e:
            raise RuntimeError(
                "Failed to give user name while logging into Padlet"
                f" platform. {e}"
            ) from None

        # Click the continue button
        try:
            selector = (By.CSS_SELECTOR, "button[data-pw='continueButton']")
            condition = EC.element_to_be_clickable(selector)
            btn = WebDriverWait(self._browser, 5).until(condition)
            btn.click()
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)
        except Exception as e:
            raise RuntimeError(
                "Failed to click continue button after entering "
                f"username during Padlet platform login. {e}"
            ) from None

        # Fill out the password input box
        try:
            selector = (By.CSS_SELECTOR, "input[data-pw='passwordInput']")
            condition = EC.element_to_be_clickable(selector)
            input = WebDriverWait(self._browser, 15).until(condition)
            input.send_keys(self._password)
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_ACTION)
        except Exception as e:
            raise RuntimeError(
                "Failed to give user password while logging into Padlet"
                f" platform. {e}"
            )

        # Click the final "Log in" button
        try:
            selector = (By.CSS_SELECTOR, "button[data-pw='loginButton']")
            condition = EC.element_to_be_clickable(selector)
            btn = WebDriverWait(self._browser, 5).until(condition)
            btn.click()
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)
        except Exception as e:
            raise RuntimeError(
                'Failed to click final "Log in" button after entering '
                f"username and password on Padlet platform. {e}"
            ) from None

    def _retry_action(
        self, func: Callable, error_msg: str, num_retries: int = 1
    ) -> None:
        """Retries a WebDriver action up to a given
        number of times before raising an exception.

        Raises:
            `RuntimeError` if the action has still not been
                completed successfully after exhausting all retries.

        Args:
            func (`Callable`): The action to attempt.

            error_msg (`str`): The message to display if the action fails.

            num_retries (`int`): The number of times to reattempt
                the action. Defaults to one.

        Returns:
            `None`
        """
        # Initialize variables
        retries = 0
        last_error = None

        # Enter retry loop
        while True:
            try:
                # Raise exception if no more retries remaining
                if retries > num_retries:
                    raise RuntimeError(f"{error_msg} {last_error}")

                # Attempt action and return upon success
                return func()

            except Exception as e:
                # If failure occurs, log warning and wait for new page refresh
                self._logger.warning(
                    f"Warning - {error_msg} Refreshing page and retrying after"
                    " waiting for"
                    f" {self.DEFAULT_SECONDS_WAIT_PAGE_LOAD} second(s). Error"
                    f" message: {e}."
                )
                self._browser.refresh()
                time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)

                # Ierate the number of retries
                last_error = str(e)
                retries += 1

    def add_board(self) -> str:
        """Creates a new Padlet board from a pre-configured template.
        Retries have been configured for WebDriver actions to account
        for periodic popups and surveys that appear on the site and
        intercept clicks.

        Args:
            `None`

        Returns:
            (`str`): The id of the newly-created board.
        """

        # Define local function to click "Make a Padlet" button
        def click_make_button():
            selector = (
                By.CSS_SELECTOR,
                "a[data-testid='dashMakePadletButton']",
            )
            condition = EC.element_to_be_clickable(selector)
            btn = WebDriverWait(self._browser, 5).until(condition)
            btn.click()
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_ACTION)

        # Attempt click with retries
        err_msg = "Failed to click the button to create a new Padlet board."
        self._retry_action(click_make_button, err_msg)

        # Define local function to use the only template available to create a board
        def click_template():
            selector = (
                By.CSS_SELECTOR,
                "button[data-testid='MapLayoutPickerWallCard']",
            )
            condition = EC.element_to_be_clickable(selector)
            btn = WebDriverWait(self._browser, 5).until(condition)
            btn.click()
            time.sleep(self.DEFAULT_SECONDS_WAIT_PAGE_LOAD)

        # Attempt click with retries
        err_msg = (
            f"Failed to click on the Padlet template to use as the basis of the"
            f" map."
        )
        self._retry_action(click_template, err_msg)

        # Parse id of new board from current URL
        board_id = self._browser.current_url.split("-")[-1]

        return board_id

    def add_post(self, board_id: str, post: Dict) -> Dict:
        """Adds a new post to an existing board.

        Documentation:
        - ["Padlet Developers - Create a post"](https://docs.padlet.dev/reference/add-post)

        Args:
            board_id (`str`): The unique identifier for the board.

            post (`dict`): The post request.

        Returns:
            (`dict`): The newly-created post.
        """
        url = f"https://api.padlet.dev/v1/boards/{board_id}/posts"
        headers = {"X-Api-Key": self._api_key}
        r = requests.post(url, headers=headers, json=post)
        if not r.ok:
            raise RuntimeError(
                f"Failed to create new post {str(post)}."
                f'The response returned a "{r.status_code} - {r.reason}" '
                f'status code with the message "{r.text}".'
            )
        return r.json()

    def get_board(self, board_id: str) -> Dict:
        """Fetches an existing board by id.

        Documentation:
        - ["Padlet Developers - Get board by id"](https://docs.padlet.dev/reference/get-board-by-id)

        Args:
            board_id (`str`): The unique identifier for the board.

        Returns:
            (`dict`): A representation of the board,
                with all post and section data.
        """
        url = f"https://api.padlet.dev/v1/boards/{board_id}?include=posts,sections"
        headers = {"X-Api-Key": self._api_key}
        r = requests.get(url, headers=headers)
        if not r.ok:
            raise RuntimeError(
                f'Failed to retrieve Padlet board "{board_id}".'
                f'The response returned a "{r.status_code} - {r.reason}" '
                f'status code with the message "{r.text}".'
            )
        return r.json()

    def add_board_with_posts(self, locations: List[Dict]) -> Dict:
        """Creates a new Padlet board formatted as a map and
        adds posts representing indoor and outdoor bin locations.

        Args:
            locations (`list` of `dict`): A list of locations to map.

        Returns:
            (`dict`): A representation of the newly-created board,
                with all post and section data.
        """
        # Initialize web browser instance
        self._initialize_browser()

        # Create new board from template
        self._logger.info(
            "Creating new Padlet board from pre-configured template."
        )
        board_id = self.add_board()
        self._logger.info("Board successfully created.")

        # Get board metadata
        self._logger.info("Fetching board metadata from Padlet API.")
        board_metadata = self.get_board(board_id)
        self._logger.info("Metadata successfully retrieved.")

        # Parse ids of custom board sections (indoor and outdoor bins)
        self._logger.info("Parsing board metadata for section ids.")
        indoor_section_id = None
        outdoor_section_id = None
        for obj in board_metadata["included"]:
            if obj["type"] != "section":
                continue
            elif obj["attributes"]["title"] == self._indoor_bins_section:
                indoor_section_id = obj["id"]
            elif obj["attributes"]["title"] == self._outdoor_bins_section:
                outdoor_section_id = obj["id"]
            else:
                raise RuntimeError(
                    f'An unexpected section, "{obj["attributes"]["title"]}",'
                    " was encountered. Please confirm that only the sections"
                    f' "{self._indoor_bins_section}" and'
                    f' "{self._outdoor_bins_section}"exist in the Padlet'
                    " template."
                )

        # Add posts to board one at a time, as permitted by API
        self._logger.info(
            "Adding posts to board one at a time, as permitted through API."
        )
        for location in locations:
            # Add slight delay in between requests to avoid throttling
            time.sleep(self.SECONDS_DELAY_PER_REQUEST)

            # Initialize request body
            post = {
                "data": {
                    "type": "post",
                    "attributes": {
                        "content": {
                            "subject": location["name"],
                            "body": (
                                f"Source: {location['source']}\n"
                                f"Categories: {location['categories']}"
                            ),
                        },
                        "mapProps": {
                            "longitude": location["lon"],
                            "latitude": location["lat"],
                            "locationName": location["address"],
                        },
                    },
                    "relationships": {
                        "section": {"data": {"id": outdoor_section_id}}
                    },
                }
            }

            # Add attachment object if location has url
            if location["url"]:
                # Parse URL scheme
                scheme = urlparse(location["url"]).scheme

                # Assign default scheme if none found
                url = (
                    f"https://{location['url']}"
                    if not scheme
                    else location["url"]
                )

                # Add attachment object
                post["data"]["attributes"]["content"]["attachment"] = {
                    "url": url,
                    "caption": "Website Url",
                }

            # Add post to board
            self.add_post(board_id, post)

        # Log total number of posts uploaded to board
        self._logger.info(f"{len(locations)} location(s) added successfully.")

        # Return final board metadata, which includes new posts
        self._logger.info("Fetching metadata for final state of board.")
        return self.get_board(board_id)
