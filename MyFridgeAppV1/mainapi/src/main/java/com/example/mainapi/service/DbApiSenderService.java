package com.example.mainapi.service;

import com.example.mainapi.model.ProductTransferDto;
import com.example.mainapi.model.Recipe;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;

@Service
public class DbApiSenderService 
{

    private final RestTemplate restTemplate = new RestTemplate();

    public void sendProductToDb(ProductTransferDto dto) 
    {
        String url = "http://localhost:8082/api/fridge/add";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<ProductTransferDto> request = new HttpEntity<>(dto, headers);

        try 
        {
            ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
            System.out.println("DBAPI'ye gönderilen ürün yanıtı: " + response.getBody());
        } 
        catch (Exception e) 
        {
            System.err.println("Ürün ekleme sırasında hata: " + e.getMessage());
        }
    }

    public boolean deleteProductFromFridge(Long userId, String productName) 
    {
        try 
        {
            String url = "http://localhost:8082/api/fridge/delete?userId=" + userId + "&productName=" + productName;
            ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, null, String.class);
            return response.getStatusCode().is2xxSuccessful();
        } 
        catch (Exception e) 
        {
            System.err.println("DBAPI silme isteği başarısız: " + e.getMessage());
            return false;
        }
    }

    public void sendRecipeToDb(Recipe recipe, Long userId) 
    {
        String url = "http://localhost:8082/api/recipe/add/" + userId;
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Recipe> request = new HttpEntity<>(recipe, headers);

        try 
        {
            ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
            System.out.println("Tarif ekleme yanıtı: " + response.getBody());
        } 
        catch (Exception e) 
        {
            System.err.println("Tarif gönderme hatası: " + e.getMessage());
        }
    }

    public void eksiltUserFridge(Long userId, String productName, int miktar) 
    {
        String url = "http://localhost:8082/api/fridge/eksilt";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_FORM_URLENCODED);

        MultiValueMap<String, String> params = new LinkedMultiValueMap<>();
        params.add("userId", userId.toString());
        params.add("productName", productName);
        params.add("miktar", String.valueOf(miktar));

        HttpEntity<MultiValueMap<String, String>> request = new HttpEntity<>(params, headers);

        try 
        {
            ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);
            System.out.println("Eksiltme yanıtı: " + response.getBody());
        } 
        catch (Exception e) 
        {
            System.err.println("Eksiltme işlemi sırasında hata: " + e.getMessage());
        }
    }

}
