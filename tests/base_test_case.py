import unittest
import logging

# Suppress logging during tests to keep output clean
logging.disable(logging.CRITICAL)


class BaseTestCase(unittest.TestCase):
    """Base class for all test cases in Project Chimera."""

    @classmethod
    def setUpClass(cls):
        """Set up for all test cases in this class."""
        super().setUpClass()
        # Add any common setup logic here that runs once per test class
        # For example, initializing a shared mock or resource
        # print(f"\nSetting up test class: {cls.__name__}...")

    @classmethod
    def tearDownClass(cls):
        """Tear down for all test cases in this class."""
        # Add any common cleanup logic here that runs once per test class
        # print(f"Tearing down test class: {cls.__name__}.")
        super().tearDownClass()

    def setUp(self):
        """Set up for each individual test method."""
        super().setUp()
        # Add any common setup logic here, e.g., initializing shared mocks or resources
        # print(f"  Setting up test method: {self._testMethodName}")

    def tearDown(self):
        """Tear down for each individual test method."""
        # Add any common cleanup logic here
        # print(f"  Tearing down test method: {self._testMethodName}")
        super().tearDown()
