# Personal Resume Website

## Adding Your Profile Photo

To add your actual profile photo:

1. Save your photo as `profile.jpg` (or .png) in the root directory
2. Update line 24 in `index.html`:
   ```html
   <img src="profile.jpg" alt="Dain Kim">
   ```

Alternatively, you can:
- Use a URL from LinkedIn or another source
- Create an `assets/images/` folder and place your photo there

The photo will be automatically styled as a circle with a subtle border.

## Adding Your CV

To add your CV for download:

1. Save your CV as `cv.pdf` in the root directory
2. The link is already configured in the HTML

Alternatively, you can:
- Use a different filename and update line 35 in `index.html`
- Host the CV on Google Drive or another service and use the external URL

## Deployment

This site is configured for GitHub Pages. Any push to the main branch will automatically deploy to:
https://dain5832.github.io/

## Customization

- Edit `index.html` to update content
- Modify CSS variables in `styles.css` to change colors
- Navigation and responsive design are already configured