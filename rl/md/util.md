### 概念
實作「安全除法」與「安全對數」的函數，避免因為分母為零或輸入值為零而引發錯誤（例如 NaN 或無窮大）


### 1. `safe_div` 函數

這個函數進行安全的除法運算，並在分母為零的情況下返回零

* 參數
  * `numerator`：任意張量，代表分子的值
  * `denominator`：與 `numerator` 的形狀相同，值假設為非負數的張量，代表分母
  * `name`：返回的操作名稱（選填，預設為 `"value"`）
    
* 功能
  1. 檢查 `denominator` 是否大於 0。如果條件成立，執行真正的除法運算
  2. 為避免分母為零的問題，將分母中為零的元素替換為與其形狀相同的全為 1 的張量
  3. 如果 `denominator` 小於或等於 0，則直接返回一個與 `numerator` 形狀相同的零張量
     
* 程式邏輯
  * 使用 `tf.where` 判斷分母是否大於 0
  * 進一步使用 `tf.div`（已過時，建議改用`tf.divide`）進行實際的除法操作

### 2. `safe_log` 函數

這個函數計算安全的對數，當輸入值為零時返回零

* 參數
  * `x`：任意張量，代表輸入的數值
* 功能
  1. 檢查輸入值是否為零。如果值為零，直接返回零
  2. 為避免對零或負數取對數的錯誤，將輸入值與一個極小的正數（`1e-12`）取最大值，然後再取對數
* 程式邏輯
  * 使用 `tf.where` 判斷輸入值是否為零。
  * 如果值為零，返回一個與輸入張量形狀相同的零張量
  * 否則計算對數（取最大值以避免非法操作）
    
### 程式
```
import tensorflow as tf

def safe_div(numerator, denominator, name="value"):
  """Computes a safe divide which returns 0 if the denominator is zero.
  Note that the function contains an additional conditional check that is
  necessary for avoiding situations where the loss is zero causing NaNs to
  creep into the gradient computation.
  Args:
    numerator: An arbitrary `Tensor`.
    denominator: `Tensor` whose shape matches `numerator` and whose values are
      assumed to be non-negative.
    name: An optional name for the returned op.
  Returns:
    The element-wise value of the numerator divided by the denominator.
  """

#------------------------------------------------------------------------------------
# safe_div 函數
  return tf.where(
      tf.greater(denominator, 0),  # 檢查分母是否大於 0
      tf.div(numerator, tf.where(  # 分母不為零時計算分子除以分母
          tf.equal(denominator, 0),  # 若分母為 0
          tf.ones_like(denominator),   # 替換分母為 1
          denominator)),  # 否則保持原值
      tf.zeros_like(numerator),  # 分母為零時返回 0
      name=name)  # 設定操作名稱

#------------------------------------------------------------------------------------
# safe_log 函數
def safe_log(x):
  """Computes a safe logarithm which returns 0 if x is zero."""
  return tf.where(
      tf.equal(x, 0),  # 檢查 x 是否為 0
      tf.zeros_like(x),  # 如果是 0，返回 0
      tf.log(tf.maximum(1e-12, x)))  # 否則計算 log，取最大值以防止錯誤
```
### 總結
1. 處理零分母：`safe_div` 確保分母不會為零，避免 NaN 的出現
2. 處理零或負對數：`safe_log` 以極小的正數代替零或負值，確保對數計算的穩定性
3. 數值穩定性：這些函數在機器學習中很重要，特別是在計算損失函數時，可避免數值溢出或不穩定
