package com.example.OCR_API.controller;
import com.example.OCR_API.service.OcrSenderService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("/ocr")
public class OcrTestController 
{

    private final OcrSenderService ocrSenderService;

    public OcrTestController(OcrSenderService ocrSenderService) 
    {
        this.ocrSenderService = ocrSenderService;
    }

    @GetMapping("/send-latest")
    public ResponseEntity<String> sendLatest() 
    {
        ocrSenderService.sendLatestParsedDataToMainApi();
        return ResponseEntity.ok("Son JSON dosyası MainAPI'ye gönderildi.");
    }
}
