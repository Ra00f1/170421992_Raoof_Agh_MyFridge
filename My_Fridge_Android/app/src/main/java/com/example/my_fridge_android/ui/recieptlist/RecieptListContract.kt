package com.example.my_fridge_android.ui.recieptlist

import com.example.my_fridge_android.data.source.remote.FridgeItem

object RecieptListContract {
    data class UiState(
        val isLoading: Boolean = false,
        val fridgeItems: List<FridgeItem> = emptyList(),
        val showImagePickerSheet: Boolean = false,
        val showAddIngredientsSheet: Boolean = false,
        val uploadStatus: String? = null,
        val errorMessage: String? = null,
        val isDeleting: Boolean = false,
        val deletingItemId: String? = null,
        val isCameraLaunching: Boolean = false
    )

    sealed class UiAction {
        object NavigateToHome : UiAction()
        object ShowImagePicker : UiAction()
        object HideImagePicker : UiAction()
        object TakePhoto : UiAction()
        object SelectFromGallery : UiAction()
        object TestServerConnection : UiAction()
        object ClearUploadStatus : UiAction()
        object LoadFridgeItems : UiAction()
        data class DeleteReceipt(val receiptId: String) : UiAction()
        object ShowAddIngredients : UiAction()
        object HideAddIngredients : UiAction()
        data class AddIngredient(
            val name: String,
            val amount: String,
            val amountType: String
        ) : UiAction()
        data class IncreaseItemAmount(val itemId: Int) : UiAction()
        data class DecreaseItemAmount(val itemId: Int) : UiAction()
    }

    sealed class UiEffect {
        object NavigateToHome : UiEffect()
        object OpenCamera : UiEffect()
        object OpenGallery : UiEffect()
        data class ShowError(val message: String) : UiEffect()
        data class ShowMessage(val message: String) : UiEffect()
    }
}
