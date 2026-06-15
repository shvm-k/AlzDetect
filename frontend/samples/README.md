# Sample MRI scans (optional)

The frontend shows **"Try a sample"** buttons under the upload box. Each button
loads an image from this folder and sends it through the live model — great for
letting reviewers/professors test the app with one click, no MRI file needed.

The buttons **auto-hide** if the matching file is missing, so the page never
shows a broken button. To enable them, drop in JPGs named exactly:

| Button     | File           | Pick a slice that is…              |
|------------|----------------|------------------------------------|
| HEALTHY    | `healthy.jpg`  | a clear Non-Demented axial slice   |
| VERY MILD  | `verymild.jpg` | a Very mild Dementia slice         |
| MILD       | `mild.jpg`     | a Mild Dementia slice              |
| MODERATE   | `moderate.jpg` | a Moderate Dementia slice          |

How to get them: copy four representative slices from the training dataset
(`legendahmed/alzheimermridataset`, the `all image/` folder — files are prefixed
`nonDem*`, `verymildDem*`, `mildDem*`, `moderateDem*`), rename them as above, and
place them here. Any size works (the model resizes to 224×224); these are just
for the demo buttons.

> Tip: pick slices the model classifies *correctly* so the demo is convincing,
> and avoid using exact training images if you want an honest demo — a held-out
> slice is fairer.
