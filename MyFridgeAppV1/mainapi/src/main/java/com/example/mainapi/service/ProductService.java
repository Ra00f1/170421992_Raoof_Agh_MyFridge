package com.example.mainapi.service;

import com.example.mainapi.util.FileWriterUtil;
import com.example.mainapi.dto.OcrProductDto;
import com.example.mainapi.model.ProductTransferDto;

import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.List;

@Service
public class ProductService 
{

    private final DbApiSenderService dbApiSenderService;

    public ProductService(FileWriterUtil fileWriterUtil, DbApiSenderService dbApiSenderService) 
    {
        this.dbApiSenderService = dbApiSenderService;
    }

    public void processOcrProducts(OcrProductDto dto) 
    {
        Long userId = dto.getUserid();
        List<List<String>> items = dto.getItems();

        for (List<String> item : items) 
        {
            try 
            {
                String name = item.get(0).trim().toLowerCase();
                int amount = Integer.parseInt(item.get(1).trim());
                String countType = item.get(2).trim().toUpperCase();
                double price = Double.parseDouble(item.get(3).replaceAll("[^\\d,\\.]", "").replace(",", "."));
                String skt = item.get(4).trim();

                String category = extractCategory(name);

                ProductTransferDto transferDto = new ProductTransferDto();
                transferDto.setProductName(name);
                transferDto.setCategoryName(category);
                transferDto.setPrice(price);
                transferDto.setAmount(amount);
                transferDto.setCountType(countType);
                transferDto.setSkt(skt);
                transferDto.setUserId(userId);

                dbApiSenderService.sendProductToDb(transferDto);
            } catch (Exception e) {
                System.err.println("Hatalı veri: " + item + " => " + e.getMessage());
            }
        }
    }
    private String extractCategory(String name) 
    {
        String lower = name.toLowerCase();
        if (lower.contains("yoğurt")) return "yoğurt";
        if (lower.contains("cola")) return "cola";
        if (lower.contains("pirinç")) return "pirinç";
        return "diğer";
    }

    public boolean deleteProductFromFridge(Long userId, String productName) 
    {
        try 
        {
            String url = "http://localhost:8082/api/fridge/delete?userId=" + userId + "&productName=" + productName;
            RestTemplate restTemplate = new RestTemplate();
            ResponseEntity<String> response = restTemplate.exchange(
                    url, HttpMethod.DELETE, null, String.class);

            return response.getStatusCode().is2xxSuccessful();
        } 
        catch (Exception e) 
        {
            System.err.println("DBAPI silme isteği başarısız: " + e.getMessage());
            return false;
        }
    }

}
