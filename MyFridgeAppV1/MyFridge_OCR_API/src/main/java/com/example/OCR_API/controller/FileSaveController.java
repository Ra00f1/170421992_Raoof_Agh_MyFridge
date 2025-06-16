package com.example.OCR_API.controller;

import com.example.OCR_API.model.OcrPayload;
import com.example.OCR_API.service.OcrSenderService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@RestController
@RequestMapping("/api/ocr")
public class FileSaveController 
{

    @Value("${save.directory}")
    private String saveDir;

    private final OcrSenderService ocrSenderService;

    public FileSaveController(OcrSenderService ocrSenderService) {
        this.ocrSenderService = ocrSenderService;
    }

    @PostMapping("/convert")
    public ResponseEntity<String> convertAndSave(@RequestBody OcrPayload payload) 
    {
        try 
        {
            int userId = payload.getUserid();
            List<List<String>> items = payload.getItems();

            if (items == null || items.isEmpty()) 
            {
                return ResponseEntity.badRequest().body("items boş olamaz.");
            }

            List<List<String>> formattedItems = new ArrayList<>();

            for (List<String> item : items) 
            {
                if (item.size() < 5) continue;

                String name = clean(item.get(0));
                String amount = clean(item.get(1));
                String unit = clean(item.get(2));
                String price = clean(item.get(3));
                String skt = clean(item.get(4));

                formattedItems.add(Arrays.asList(name, amount, unit, price, skt));
            }

            // Yeni payload nesnesi oluştur
            OcrPayload formattedPayload = new OcrPayload();
            formattedPayload.setUserid(userId);
            formattedPayload.setItems(formattedItems);

            // Dosya oluştur
            Files.createDirectories(Paths.get(saveDir));
            String fileName = "ocr_" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")) + ".json";
            Path filePath = Paths.get(saveDir, fileName);

            ObjectMapper mapper = new ObjectMapper();
            mapper.writerWithDefaultPrettyPrinter().writeValue(filePath.toFile(), formattedPayload);

            // MainAPI'ye gönder
            ocrSenderService.sendLatestParsedDataToMainApi();

            return ResponseEntity.ok("JSON dosyası oluşturuldu ve gönderildi: " + fileName);
        } 
        catch (Exception e) 
        {
            return ResponseEntity.internalServerError().body("Hata: " + e.getMessage());
        }
    }

    private String clean(String s) 
    {
        return s == null ? "" : s.trim().replaceAll("[^A-ZÇĞİÖŞÜa-zçğıöşü0-9,\\.\\-\\/]", "").replace(",", ".");
    }
}
