import urllib.request as req
from dataclasses import dataclass
import json


@dataclass
class Product:
    id: str
    name: str
    review_count: int | None
    rating_value: int | None
    price: int
    price_z_score: float | None = None


class PChomeScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.products = []

    def fetch_data(self):
        page = 1
        while True:
            request = req.Request(f"{self.base_url}&page={page}")
            with req.urlopen(request) as response:
                response = json.loads(response.read().decode("utf-8"))

                products = []
                for product in response["Prods"]:
                    self.products.append(
                        Product(
                            id=product["Id"],
                            name=product["Name"],
                            review_count=product["reviewCount"],
                            rating_value=product["ratingValue"],
                            price=product["Price"],
                        )
                    )
                if page >= response["TotalPage"]:
                    break
                page += 1

    def export_product_ids(self):
        with open("products.txt", "w") as file:
            for product in self.products:
                file.write(f"{product.id}\n")

    def export_best_products(self):
        filtered_product_ids = [
            prod.id
            for prod in self.products
            if prod.review_count is not None
            and prod.review_count > 0
            and prod.rating_value is not None
            and prod.rating_value > 4.9
        ]

        # Write the filtered product IDs to a file
        with open("best-products.txt", "w") as file:
            for product_id in filtered_product_ids:
                file.write(f"{product_id}\n")

    def calculate_average_price_products_by_i5(self):
        i5_products = [p for p in self.products if "i5" in p.name]
        if i5_products:
            average_price = sum(p.price for p in i5_products) / len(i5_products)
            print(
                f"The average price of ASUS PCs with Intel i5 processor is {average_price:.2f} TWD"
            )
        else:
            print("No ASUS PCs with Intel i5 processor found.")

    def export_z_scores(self):
        # Extract prices
        prices = [product.price for product in self.products]

        # Calculate mean price
        mean_price = sum(prices) / len(prices)

        # Calculate standard deviation
        variance = sum((price - mean_price) ** 2 for price in prices) / len(prices)
        std_deviation = variance**0.5

        # Calculate z-scores and add to product data
        for product in self.products:
            z_score = (product.price - mean_price) / std_deviation
            product.price_z_score = z_score

        # Write to standardization.csv
        with open("standardization.csv", "w") as file:
            # Write header
            file.write("ProductID,Price,PriceZScore\n")
            # Write product data
            for product in self.products:
                file.write(
                    f"{product.id},{product.price},{product.price_z_score:.4f}\n"
                )

        print("standardization.csv has been created.")


def main():
    scraper = PChomeScraper(
        base_url="https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&attr=&pageCount=40"
    )
    scraper.fetch_data()
    scraper.export_product_ids()
    scraper.export_best_products()
    scraper.calculate_average_price_products_by_i5()
    scraper.export_z_scores()


if __name__ == "__main__":
    main()
