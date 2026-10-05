import { expect, test } from "@playwright/test";

test("browse fixture pages and segments while labelling", async ({ page }) => {
  await page.goto("/label");
  const city = page.getByTestId("label-city");
  await expect(city.locator("option")).toHaveText(["Select city", "Alder", "Birch"]);
  await city.selectOption({ label: "Birch" });
  const document = page.getByTestId("label-document");
  await expect(document.locator('option[value="birch-minutes-201"]')).toBeAttached();
  await document.selectOption("birch-minutes-201");
  await expect(page.getByTestId("page-image")).toBeVisible();
  await expect(page.getByTestId("unit-text")).toBeVisible();
  await page.getByTestId("unit-next").click();
  await expect(page.getByTestId("unit-label")).toHaveText("p. 2 of 3");
  await page.getByTestId("unit-next").click();
  await expect(page.getByTestId("unit-label")).toHaveText("p. 3 of 3");
  await expect(page.getByTestId("unit-text")).toContainText("ORDINANCE 1042");
  await document.selectOption("birch-transcript-201");
  await expect(page.getByTestId("segment-time")).toBeVisible();
  await expect(page.getByTestId("page-image")).toHaveCount(0);
  for (const id of ["results", "answer", "facts-table"]) {
    await expect(page.getByTestId(id)).toHaveCount(0);
  }
});
