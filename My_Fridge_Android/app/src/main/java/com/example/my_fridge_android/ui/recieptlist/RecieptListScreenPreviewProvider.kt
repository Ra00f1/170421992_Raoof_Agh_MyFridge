package com.example.my_fridge_android.ui.recieptlist

import androidx.compose.ui.tooling.preview.PreviewParameterProvider
import com.example.my_fridge_android.data.source.remote.FridgeItem
import com.example.my_fridge_android.data.source.remote.Product

class RecieptListScreenPreviewProvider : PreviewParameterProvider<RecieptListContract.UiState> {
    override val values: Sequence<RecieptListContract.UiState>
        get() = sequenceOf(
            RecieptListContract.UiState(
                isLoading = true,
                fridgeItems = emptyList(),
                showImagePickerSheet = false,
                uploadStatus = null,
            ),
            RecieptListContract.UiState(
                isLoading = false,
                fridgeItems = emptyList(),
                showImagePickerSheet = false,
                uploadStatus = null,
            ),
            RecieptListContract.UiState(
                isLoading = false,
                fridgeItems = listOf(
                    FridgeItem(
                        id = 1,
                        userId = 1,
                        amount = 2,
                        eklenmeTarihi = "2025-06-14",
                        skt = "2025-07-01",
                        product = Product(
                            id = 1,
                            name = "Sütaş Yoğurt",
                            category = "Süt Ürünleri",
                            price = 29.5,
                            countType = "adet"
                        )
                    ),
                    FridgeItem(
                        id = 2,
                        userId = 1,
                        amount = 4,
                        eklenmeTarihi = "2025-06-14",
                        skt = "2025-07-01",
                        product = Product(
                            id = 2,
                            name = "Coca Cola Kola",
                            category = "İçecek",
                            price = 25.0,
                            countType = "Litre"
                        )
                    )
                ),
                showImagePickerSheet = false,
                uploadStatus = null,
            ),
            RecieptListContract.UiState(
                isLoading = false,
                fridgeItems = emptyList(),
                showImagePickerSheet = true,
                uploadStatus = null,
            ),
        )
}
