import { expect, test } from "@playwright/test";

const NAV = [
  ["nav-search", "Search"],
  ["nav-facts", "Facts"],
  ["nav-ask", "Ask"],
  ["nav-agent", "Agent"],
  ["nav-label", "Label"],
];

test("search the fixture corpus filtered by city", async ({ page }) => {
  await page.goto("/");
  for (const [id, text] of NAV) {
    await expect(page.getByTestId(id)).toHaveText(text);
  }
  const city = page.getByTestId("filter-city");
  await expect(city.locator("option")).toHaveText(["All cities", "Alder", "Birch"]);

  const input = page.getByTestId("search-input");
  await input.fill("sidewalk repair");
  await city.selectOption({ label: "Birch" });
  await input.press("Enter");

  await expect(page.getByTestId("result-1")).toBeVisible();
  const headings = await page.getByTestId("result-heading").allTextContents();
  expect(headings.length).toBeGreaterThan(0);
  for (const heading of headings) {
    expect(heading.startsWith("Birch")).toBe(true);
  }
});
