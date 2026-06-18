wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.body = [[{
  "checking_status":"<0","duration":24.0,"credit_history":"existing paid",
  "purpose":"radio/tv","credit_amount":3500.0,"savings_status":"<100",
  "employment":"1<=X<4","installment_commitment":4.0,"personal_status":"male single",
  "other_parties":"none","residence_since":2.0,"property_magnitude":"car","age":30.0,
  "other_payment_plans":"none","housing":"own","existing_credits":1.0,"job":"skilled",
  "num_dependents":1.0,"own_telephone":"none","foreign_worker":"yes"
}]]
request = function() return wrk.format(nil, "/predict") end
