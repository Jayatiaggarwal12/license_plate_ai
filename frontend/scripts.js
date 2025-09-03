document.getElementById('imageInput').addEventListener('change', handleImageUpload);

async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    // Display original image
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('originalImage').src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Call API
        const response = await fetch('/detect-license-plate', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        if (result.success) {
            // Show results
            document.getElementById('results').style.display = 'block';
            document.getElementById('confidence').textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
            document.getElementById('licensePlate').textContent = result.license_plate;
            document.getElementById('enhancedImage').src = result.enhanced_image;
            document.getElementById('rawOcr').textContent = result.raw_ocr;
            document.getElementById('correctedText').textContent = result.license_plate;
        } else {
            alert(result.message);
        }
        
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        alert('Error processing image: ' + error.message);
    }
}